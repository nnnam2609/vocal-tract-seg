"""
MedSAM2 Inference + Evaluation Script (config-driven)
Generates evaluation_results_detailed.csv with the same structure as YOLO evaluation.
Uses YOLO-format labels as ground truth and MedSAM2 predictions (box prompts from GT).
"""

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import binary_fill_holes

import torch

# Repo root (for vt_tools / vt_tracker imports)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))
sys.path.insert(0, str(REPO_ROOT.parent / "vt_tools"))
sys.path.insert(0, str(REPO_ROOT.parent / "vt_tracker"))

# MedSAM2 imports (from external/MedSAM2)
MEDSAM2_ROOT = REPO_ROOT / "external" / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# vt_tools / vt_tracker (same post-processing as YOLO/MaskRCNN)
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker.postprocessing import POST_PROCESSING
from vt_tracker.postprocessing.calculate_contours import calculate_contour


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = config_path
    config["_config_dir"] = os.path.dirname(os.path.abspath(config_path))
    return config


def resolve_path(path_value, config_dir, repo_root):
    if not path_value:
        return path_value
    path_str = str(path_value)
    if os.path.isabs(path_str):
        return path_str

    candidate = os.path.abspath(os.path.join(config_dir, path_str))
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.abspath(os.path.join(repo_root, path_str))
    return candidate


def parse_image_name(image_name):
    base_name = os.path.splitext(image_name)[0]
    parts = base_name.split("_")
    subject = None
    sequence = None
    frame = None

    try:
        if len(parts) >= 6 and parts[1] == "Database":
            subject = parts[3]
            sequence = parts[4]
            frame = int(parts[5])
        elif len(parts) >= 7 and parts[1] == "Vocal" and parts[2] == "Tract":
            subject = parts[4]
            sequence = parts[5]
            frame = int(parts[6])
        else:
            for i in range(len(parts) - 2):
                try:
                    if parts[i + 1].startswith("S") and parts[i].isdigit() and parts[i + 2].isdigit():
                        subject = parts[i]
                        sequence = parts[i + 1]
                        frame = int(parts[i + 2])
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"Warning: Could not parse image name '{image_name}': {e}")

    return subject, sequence, frame


def get_dataset_name(image_name):
    base_name = os.path.splitext(image_name)[0]
    parts = base_name.split("_")
    if len(parts) >= 6 and parts[1] == "Database":
        return "_".join(parts[:3])  # ArtSpeech_Database_2
    if len(parts) >= 7 and parts[1] == "Vocal" and parts[2] == "Tract":
        return "_".join(parts[:4])  # ArtSpeech_Vocal_Tract_Segmentation
    return "UnknownDataset"


def save_pred_contour(output_folder, dataset, subject, sequence, frame, class_name, contour):
    if subject is None or sequence is None or frame is None:
        return
    out_dir = Path(output_folder) / "inference_contours" / dataset / str(subject) / str(sequence)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_str = f"{int(frame):04d}"
    np.save(out_dir / f"{frame_str}_{class_name}.npy", contour)


def log_zero_p2cp(output_folder, image_name, class_name, result, pred_contour, gt_contour, eps=1e-6):
    p2cp_mean = result.get("p2cp_mean")
    if p2cp_mean is None or not np.isfinite(p2cp_mean) or p2cp_mean > eps:
        return
    log_path = Path(output_folder) / "p2cp_zero_debug.log"
    pred_len = len(pred_contour) if pred_contour is not None else 0
    gt_len = len(gt_contour) if gt_contour is not None else 0
    same = False
    if pred_contour is not None and gt_contour is not None:
        same = pred_contour.shape == gt_contour.shape and np.allclose(pred_contour, gt_contour)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            "image={image} class={cls} pred_pixels={pp} gt_pixels={gp} "
            "pred_len={pl} gt_len={gl} p2cp_mean={pm:.6f} p2cp_rms={pr:.6f} "
            "contours_equal={eq}\n".format(
                image=image_name,
                cls=class_name,
                pp=result.get("pred_pixels"),
                gp=result.get("gt_pixels"),
                pl=pred_len,
                gl=gt_len,
                pm=result.get("p2cp_mean", float("nan")),
                pr=result.get("p2cp_rms", float("nan")),
                eq=same,
            )
        )


def smooth_contour(contour):
    try:
        res_x, res_y = regularize_Bsplines(contour, 3)
        return np.array([res_x, res_y]).T
    except Exception:
        return contour


def log_contour_fallback(log_path, image_name, class_name, mask_kind, reason, mask_pixels):
    if not log_path:
        return
    line = (
        "image={image} class={cls} mask={mask} reason={reason} mask_pixels={pixels}\n"
        .format(
            image=image_name,
            cls=class_name,
            mask=mask_kind,
            reason=reason,
            pixels=int(mask_pixels),
        )
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def extract_contour_from_mask(
    binary_mask,
    class_name=None,
    use_vt_tracker=True,
    gravity_curve=None,
    fallback_log_path=None,
    image_name=None,
    mask_kind=None,
):
    if use_vt_tracker and class_name is not None:
        try:
            mask_normalized = binary_mask.astype(np.float32) / 255.0
            post_proc_cfg = deepcopy(POST_PROCESSING.get(class_name, {}))
            contour = calculate_contour(
                class_name,
                mask_normalized,
                gravity_curve=gravity_curve,
                cfg=post_proc_cfg,
            )
            if contour is not None and len(contour) > 0:
                contour = smooth_contour(contour)
                return contour
            log_contour_fallback(
                fallback_log_path,
                image_name,
                class_name,
                mask_kind,
                "vt_tracker_empty",
                (binary_mask > 0).sum(),
            )
        except Exception as e:
            log_contour_fallback(
                fallback_log_path,
                image_name,
                class_name,
                mask_kind,
                f"vt_tracker_exception:{e}",
                (binary_mask > 0).sum(),
            )
            print(f"    Warning: vt_tracker failed for {class_name}, falling back to OpenCV: {e}")

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea).squeeze()
    if len(contour.shape) == 1:
        return None
    contour_array = np.array([[pt[0], pt[1]] for pt in contour])
    if contour_array[0][0] < contour_array[-1][0]:
        contour_array = np.flip(contour_array, axis=0)
    return contour_array


def point_to_curve_distance(point, curve):
    distances = np.sqrt(np.sum((curve - point) ** 2, axis=1))
    return np.min(distances)


def p2cp_mean_distance(pred_contour, gt_contour):
    distances = [point_to_curve_distance(pt, gt_contour) for pt in pred_contour]
    return np.mean(distances)


def p2cp_rms_distance(pred_contour, gt_contour):
    distances = [point_to_curve_distance(pt, gt_contour) for pt in pred_contour]
    return np.sqrt(np.mean(np.array(distances) ** 2))


def jaccard_index(pred_mask, gt_mask, eps=1e-15):
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return (intersection + eps) / (union - intersection + eps)


def create_filled_mask_from_contour(contour, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    if contour is None or len(contour) < 3:
        return mask
    contour_int = contour.astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 1)
    mask = binary_fill_holes(mask).astype(int)
    return mask


def load_ground_truth_mask(label_path, img_shape):
    if not os.path.exists(label_path):
        return None
    h, w = img_shape[:2]
    masks = {}
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            class_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            coords = coords.astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 255)
            masks[class_id] = mask
    return masks


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.int64)


def resolve_config_name(model_cfg):
    config_name = model_cfg.get("config_name")
    if config_name:
        return config_name

    config_path = model_cfg.get("config_path")
    if not config_path:
        raise ValueError("Model config must set either 'config_name' or 'config_path'.")

    # If a file path is provided, use the basename without extension.
    config_path = str(config_path)
    base = os.path.basename(config_path)
    name, _ = os.path.splitext(base)
    return name


def build_predictor(model_cfg, device):
    config_name = resolve_config_name(model_cfg)
    checkpoint_path = model_cfg["checkpoint_path"]
    model = build_sam2(config_name, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def run_medsam2_on_image(predictor, img_rgb, gt_masks, config):
    inference_cfg = config.get("inference", {})
    mask_threshold = inference_cfg.get("mask_threshold", 0.0)
    multimask_output = inference_cfg.get("multimask_output", False)

    pred_masks = {}
    predictor.set_image(img_rgb)

    for class_id, gt_mask in gt_masks.items():
        bbox = mask_to_bbox(gt_mask)
        if bbox is None:
            continue
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=multimask_output,
        )
        if masks is None or len(masks) == 0:
            continue
        if masks.ndim == 3:
            if scores is not None and len(scores) > 0:
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0
            mask = masks[best_idx]
            conf = float(scores[best_idx]) if scores is not None else 0.0
        else:
            mask = masks
            conf = float(scores[0]) if scores is not None else 0.0
        mask_bin = (mask > mask_threshold).astype(np.uint8) * 255
        pred_masks[class_id] = {"mask": mask_bin, "conf": conf}

    return pred_masks


def evaluate_from_masks(image_name, img, pred_masks, gt_masks, config):
    class_names = config["labels"]
    closed_articulators = config["evaluation"]["closed_articulators"]
    postproc_cfg = config.get("postprocessing", {})
    use_vt_tracker = postproc_cfg.get("use_vt_tracker", True)
    output_folder = resolve_path(config["paths"]["output_folder"], config["_config_dir"], REPO_ROOT)
    save_contours = config.get("evaluation", {}).get("save_contours", True)
    fallback_log_path = os.path.join(output_folder, "contour_fallback.log")
    dataset = get_dataset_name(image_name)

    results = []
    subject, sequence, frame = parse_image_name(image_name)

    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        result = {
            "image_name": image_name,
            "subject": subject,
            "sequence": sequence,
            "frame": frame,
            "class_id": class_id,
            "class_name": class_name,
            "p2cp_mean": np.nan,
            "p2cp_rms": np.nan,
            "jaccard_index": np.nan,
            "has_prediction": class_id in pred_masks,
            "has_ground_truth": class_id in gt_masks,
            "confidence": pred_masks[class_id]["conf"] if class_id in pred_masks else 0.0,
            "pred_pixels": 0,
            "gt_pixels": 0,
        }

        if class_id not in gt_masks:
            results.append(result)
            continue

        gt_mask = gt_masks[class_id]
        result["gt_pixels"] = int((gt_mask > 0).sum())

        if class_id not in pred_masks:
            results.append(result)
            continue

        pred_mask = pred_masks[class_id]["mask"]
        result["pred_pixels"] = int((pred_mask > 0).sum())

        try:
            pred_contour = extract_contour_from_mask(
                pred_mask,
                class_name=class_name if use_vt_tracker else None,
                use_vt_tracker=use_vt_tracker,
                gravity_curve=None,
                fallback_log_path=fallback_log_path if use_vt_tracker else None,
                image_name=image_name,
                mask_kind="pred",
            )
            gt_contour = extract_contour_from_mask(
                gt_mask,
                class_name=class_name if use_vt_tracker else None,
                use_vt_tracker=use_vt_tracker,
                gravity_curve=None,
                fallback_log_path=fallback_log_path if use_vt_tracker else None,
                image_name=image_name,
                mask_kind="gt",
            )

            if pred_contour is None or gt_contour is None:
                results.append(result)
                continue

            if save_contours and pred_contour is not None:
                save_pred_contour(
                    output_folder,
                    dataset,
                    subject,
                    sequence,
                    frame,
                    class_name,
                    pred_contour,
                )

            result["p2cp_mean"] = p2cp_mean_distance(pred_contour, gt_contour)
            result["p2cp_rms"] = p2cp_rms_distance(pred_contour, gt_contour)

            if class_name in closed_articulators:
                pred_filled = create_filled_mask_from_contour(pred_contour, img.shape[:2])
                gt_filled = create_filled_mask_from_contour(gt_contour, img.shape[:2])
                result["jaccard_index"] = jaccard_index(pred_filled, gt_filled)

            log_zero_p2cp(output_folder, image_name, class_name, result, pred_contour, gt_contour)
        except Exception as e:
            print(f"    Error processing {class_name}: {e}")

        results.append(result)

    return results


def visualize_results(img, pred_masks, gt_masks, output_path, config):
    import matplotlib.pyplot as plt

    viz_cfg = config.get("visualization", {})
    class_names = config["labels"]
    h, w = img.shape[:2]
    gt_linestyle = viz_cfg.get("gt_linestyle", "-")
    pred_linestyle = viz_cfg.get("pred_linestyle", "--")
    show_raw = viz_cfg.get("show_raw_segmentation", True)
    use_vt_tracker = config.get("postprocessing", {}).get("use_vt_tracker", True)

    gt_contours = {}
    pred_contours = {}

    for class_id, mask in gt_masks.items():
        class_name = class_names[class_id]
        contour = extract_contour_from_mask(
            mask, class_name=class_name if use_vt_tracker else None, use_vt_tracker=use_vt_tracker
        )
        if contour is not None:
            gt_contours[class_id] = contour

    for class_id, pred_data in pred_masks.items():
        class_name = class_names[class_id]
        mask = pred_data["mask"]
        contour = extract_contour_from_mask(
            mask, class_name=class_name if use_vt_tracker else None, use_vt_tracker=use_vt_tracker
        )
        if contour is not None:
            pred_contours[class_id] = contour

    fig, axes = plt.subplots(3, 3, figsize=viz_cfg.get("figsize", [20, 14]))
    fig.suptitle(f"MedSAM2 Segmentation Evaluation - {os.path.basename(output_path)}",
                 fontsize=16, fontweight="bold")

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    gt_composite = np.zeros((h, w), dtype=np.uint8)
    for class_id, mask in gt_masks.items():
        gt_composite[mask > 0] = class_id + 1

    axes[0, 1].imshow(img, alpha=0.6)
    axes[0, 1].imshow(gt_composite, cmap="tab10",
                      alpha=viz_cfg.get("overlay_alpha", 0.5), vmin=0, vmax=10)
    axes[0, 1].set_title("Ground Truth Overlay", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    pred_composite = np.zeros((h, w), dtype=np.uint8)
    for class_id, pred_data in pred_masks.items():
        mask = pred_data["mask"]
        pred_composite[mask > 0] = class_id + 1

    axes[0, 2].imshow(img, alpha=0.6)
    axes[0, 2].imshow(pred_composite, cmap="tab10",
                      alpha=viz_cfg.get("overlay_alpha", 0.5), vmin=0, vmax=10)
    axes[0, 2].set_title("Prediction Overlay", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    if show_raw:
        axes[1, 0].imshow(gt_composite, cmap="tab10", vmin=0, vmax=10)
        axes[1, 0].set_title("GT Raw Segmentation", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred_composite, cmap="tab10", vmin=0, vmax=10)
        axes[1, 1].set_title("Prediction Raw Segmentation", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")

        comparison_img = np.zeros((h, w, 3), dtype=np.uint8)
        comparison_img[:, :, 0] = (gt_composite > 0).astype(np.uint8) * 255
        comparison_img[:, :, 1] = (pred_composite > 0).astype(np.uint8) * 255
        axes[1, 2].imshow(img, alpha=0.5)
        axes[1, 2].imshow(comparison_img, alpha=0.6)
        axes[1, 2].set_title("Raw Mask Comparison\n(GT=Red, Pred=Green, Overlap=Yellow)",
                             fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")
    else:
        for ax in axes[1, :]:
            ax.axis("off")

    axes[2, 0].imshow(img, alpha=0.5)
    for class_id, contour in gt_contours.items():
        class_name = class_names[class_id]
        axes[2, 0].plot(
            contour[:, 0], contour[:, 1],
            linewidth=viz_cfg.get("contour_linewidth", 2.5),
            linestyle=gt_linestyle,
            label=class_name,
        )
    axes[2, 0].set_title("GT Final Contours (Solid)", fontsize=12, fontweight="bold")
    axes[2, 0].legend(loc="upper right", fontsize=7)
    axes[2, 0].axis("off")

    axes[2, 1].imshow(img, alpha=0.5)
    show_conf = viz_cfg.get("show_confidence", True)
    for class_id, contour in pred_contours.items():
        class_name = class_names[class_id]
        conf = pred_masks[class_id]["conf"]
        label = f"{class_name} ({conf:.2f})" if show_conf else class_name
        axes[2, 1].plot(
            contour[:, 0], contour[:, 1],
            linewidth=viz_cfg.get("contour_linewidth", 2.5),
            linestyle=pred_linestyle,
            label=label,
        )
    axes[2, 1].set_title("Prediction Final Contours (Dashed)", fontsize=12, fontweight="bold")
    axes[2, 1].legend(loc="upper right", fontsize=7)
    axes[2, 1].axis("off")

    axes[2, 2].imshow(img, alpha=0.5)
    for class_id, contour in gt_contours.items():
        class_name = class_names[class_id]
        axes[2, 2].plot(
            contour[:, 0], contour[:, 1],
            linewidth=2, linestyle=gt_linestyle, alpha=0.8,
            label=f"GT: {class_name}",
        )
    for class_id, contour in pred_contours.items():
        class_name = class_names[class_id]
        axes[2, 2].plot(
            contour[:, 0], contour[:, 1],
            linewidth=2.5, linestyle=pred_linestyle, alpha=0.9,
            label=f"Pred: {class_name}",
        )
    axes[2, 2].set_title("GT vs Pred Contour Overlay\n(GT=Solid, Pred=Dashed)",
                         fontsize=11, fontweight="bold")
    axes[2, 2].legend(loc="upper right", fontsize=6)
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=viz_cfg.get("dpi", 200), bbox_inches="tight", facecolor="white")
    plt.close()


def save_results(results, config):
    output_folder = config["paths"]["output_folder"]
    if not results:
        print("No results to save!")
        return

    df = pd.DataFrame(results)
    column_order = [
        "subject",
        "sequence",
        "frame",
        "image_name",
        "class_id",
        "class_name",
        "jaccard_index",
        "has_prediction",
        "has_ground_truth",
        "confidence",
        "pred_pixels",
        "gt_pixels",
    ]
    column_order = [c for c in column_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in column_order]
    df = df[column_order + remaining_cols]

    if config["evaluation"].get("save_csv", True):
        csv_path = os.path.join(output_folder, "evaluation_results_detailed.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to: {csv_path}")


def run(config_path):
    config = load_config(config_path)
    config_dir = config["_config_dir"]

    input_folder = resolve_path(config["paths"]["input_folder"], config_dir, REPO_ROOT)
    output_folder = resolve_path(config["paths"]["output_folder"], config_dir, REPO_ROOT)
    gt_folder = resolve_path(config["paths"]["ground_truth_folder"], config_dir, REPO_ROOT)

    model_cfg = config["model"]
    model_cfg["checkpoint_path"] = resolve_path(model_cfg.get("checkpoint_path"), config_dir, REPO_ROOT)

    os.makedirs(output_folder, exist_ok=True)

    if config["data"].get("process_all", True):
        image_files = sorted(
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
    else:
        list_file = config["data"].get("specific_images_file")
        if list_file:
            list_path = resolve_path(list_file, config_dir, REPO_ROOT)
            with open(list_path, "r") as f:
                image_files = [line.strip() for line in f if line.strip()]
        else:
            image_files = config["data"].get("specific_images", [])

    if not image_files:
        print("No images found to process!")
        return

    device = model_cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU.")
        device = "cpu"

    predictor = build_predictor(model_cfg, device=device)

    results_all = []
    for img_file in image_files:
        image_path = os.path.join(input_folder, img_file)
        label_path = os.path.join(gt_folder, os.path.splitext(img_file)[0] + ".txt")
        if not os.path.exists(label_path):
            print(f"⚠️  Missing label for {img_file}, skipping.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️  Could not load image {image_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_masks = load_ground_truth_mask(label_path, img.shape)
        if gt_masks is None:
            continue

        pred_masks = run_medsam2_on_image(predictor, img_rgb, gt_masks, config)

        results = evaluate_from_masks(img_file, img, pred_masks, gt_masks, config)
        results_all.extend(results)

        if config["evaluation"].get("save_visualizations", True) and config["evaluation"].get("save_per_image", True):
            viz_path = os.path.join(output_folder, f"eval_{os.path.splitext(img_file)[0]}.png")
            visualize_results(img_rgb, pred_masks, gt_masks, viz_path, config)

    save_results(results_all, config)


def main():
    parser = argparse.ArgumentParser(description="MedSAM2 inference + evaluation (config)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/Nam_exp_01082026/inference_medsam2_test.yaml",
        help="Path to MedSAM2 inference config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
