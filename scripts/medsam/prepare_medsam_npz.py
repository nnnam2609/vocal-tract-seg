#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import yaml
except Exception:
    yaml = None


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert datasets to MedSAM2 NPZ format (imgs/gts)."
    )
    parser.add_argument(
        "--source",
        choices=("yolo", "original"),
        default="yolo",
        help="Dataset source format (default: yolo).",
    )
    parser.add_argument(
        "--config",
        help="YAML config used for original-format conversion (train/val/test sequences).",
    )
    parser.add_argument(
        "--yolo-root",
        default="data_yolo",
        help="Root folder containing images/ and labels/ folders for YOLO conversion.",
    )
    parser.add_argument(
        "--output-root",
        default="data_medsam2/npz",
        help="Output folder for NPZ files (train/val/test subfolders).",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to process (default: train,val,test).",
    )
    parser.add_argument(
        "--group-by",
        choices=("sequence", "single"),
        default="sequence",
        help="(YOLO) Group frames by sequence prefix or treat each image as its own video.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include frames with no labels/masks (default: false).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="(Original) Keep frames even if some class masks are missing.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        help="Resize images/masks to fixed size (e.g., --resize 512 512).",
    )
    parser.add_argument(
        "--image-folder",
        help="Override image_folder from config (original format).",
    )
    parser.add_argument(
        "--image-ext",
        help="Override image_ext from config (original format).",
    )
    parser.add_argument(
        "--max-frames-per-seq",
        type=int,
        help="Limit frames per sequence for quick debug runs.",
    )
    parser.add_argument(
        "--exclusion-list",
        help="Path to exclusion list JSON (list of [subject, sequence, instance_number]).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a few visualizations from the prepared NPZ files.",
    )
    parser.add_argument(
        "--viz-split",
        default="train",
        help="Split to visualize (default: train).",
    )
    parser.add_argument(
        "--viz-count",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5).",
    )
    parser.add_argument(
        "--viz-out",
        default="data_medsam2/preview",
        help="Output folder for visualization images.",
    )
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=0,
        help="Random seed for visualization sampling (default: 0).",
    )
    return parser.parse_args()


def iter_images(images_dir: Path):
    for ext in IMAGE_EXTS:
        yield from images_dir.rglob(f"*{ext}")


def group_images(image_paths, group_by: str):
    groups = defaultdict(list)
    seq_re = re.compile(r"^(.*)_([0-9]+)$")

    for img_path in image_paths:
        stem = img_path.stem
        if group_by == "single":
            seq_id = stem
            frame_id = 0
        else:
            match = seq_re.match(stem)
            if match:
                seq_id = match.group(1)
                frame_id = int(match.group(2))
            else:
                seq_id = stem
                frame_id = 0
        groups[seq_id].append((frame_id, img_path))

    return groups


def load_label_polygons(label_path: Path):
    polygons = []
    if not label_path.exists():
        return polygons
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if len(poly) < 3:
                continue
            polygons.append(poly)
    return polygons


def rasterize_polygons(polygons, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint16)
    obj_id = 1
    for poly in polygons:
        pixel_poly = [(x * width, y * height) for x, y in poly]
        tmp = Image.new("L", (width, height), 0)
        ImageDraw.Draw(tmp).polygon(pixel_poly, outline=1, fill=1)
        tmp_arr = np.array(tmp, dtype=bool)
        mask[tmp_arr] = obj_id
        obj_id += 1
    return mask


def process_yolo_split(
    split: str,
    yolo_root: Path,
    output_root: Path,
    group_by: str,
    include_empty: bool,
):
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split
    if not images_dir.exists():
        print(f"[skip] missing images folder: {images_dir}")
        return

    image_paths = sorted(iter_images(images_dir))
    if not image_paths:
        print(f"[skip] no images found in: {images_dir}")
        return

    groups = group_images(image_paths, group_by=group_by)
    out_split_dir = output_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    for seq_id, frames in sorted(groups.items()):
        frames = sorted(frames, key=lambda x: x[0])
        imgs = []
        gts = []

        for _, img_path in frames:
            label_path = labels_dir / f"{img_path.stem}.txt"
            polygons = load_label_polygons(label_path)
            if not polygons and not include_empty:
                continue

            img = Image.open(img_path).convert("L")
            width, height = img.size
            mask = rasterize_polygons(polygons, width, height)

            imgs.append(np.array(img, dtype=np.uint8))
            gts.append(mask)

        if not imgs:
            continue

        imgs_arr = np.stack(imgs, axis=0)
        gts_arr = np.stack(gts, axis=0)
        out_path = out_split_dir / f"{seq_id}.npz"
        np.savez_compressed(out_path, imgs=imgs_arr, gts=gts_arr)

    print(f"[done] {split}: wrote NPZ to {out_split_dir}")


def uint16_to_uint8(img_arr: np.ndarray) -> np.ndarray:
    if img_arr.dtype == np.uint8:
        return img_arr
    max_val = float(np.max(img_arr))
    if max_val <= 0:
        return np.zeros_like(img_arr, dtype=np.uint8)
    scaled = img_arr.astype(np.float32) * (255.0 / max_val)
    return scaled.astype(np.uint8)


def equalize_hist(img_arr: np.ndarray) -> np.ndarray:
    if img_arr.dtype != np.uint8:
        img_arr = img_arr.astype(np.uint8)
    hist = np.bincount(img_arr.flatten(), minlength=256)
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    if cdf_masked.max() == cdf_masked.min():
        return img_arr
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (
        cdf_masked.max() - cdf_masked.min()
    )
    cdf_final = np.ma.filled(cdf_masked, 0).astype(np.uint8)
    return cdf_final[img_arr]


def read_frame_uint8(img_path: Path, image_ext: Optional[str]) -> np.ndarray:
    ext = (image_ext or img_path.suffix.lstrip(".")).lower()
    if ext == "npy":
        arr = np.load(img_path)
        arr = uint16_to_uint8(arr)
        arr = equalize_hist(arr)
        return arr
    if ext in ("dcm", "dicom"):
        try:
            import pydicom
        except Exception as exc:
            raise RuntimeError("pydicom is required to read DICOM files.") from exc
        ds = pydicom.dcmread(str(img_path), force=True)
        arr = uint16_to_uint8(ds.pixel_array)
        arr = equalize_hist(arr)
        return arr
    img = Image.open(img_path).convert("L")
    return np.array(img, dtype=np.uint8)


def resize_pil(img: Image.Image, size_hw: Optional[Tuple[int, int]]) -> Image.Image:
    if not size_hw:
        return img
    height, width = size_hw
    return img.resize((width, height), resample=Image.BILINEAR)


def load_rgb_triplet(
    img_path: Path,
    image_ext: str,
    size_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    stem = img_path.stem
    if stem.isdigit():
        instance_number = int(stem)
        base_dir = img_path.parent
        prev_path = base_dir / f"{instance_number - 1:04d}.{image_ext}"
        next_path = base_dir / f"{instance_number + 1:04d}.{image_ext}"
    else:
        prev_path = None
        next_path = None
    cur = read_frame_uint8(img_path, image_ext)
    prev = read_frame_uint8(prev_path, image_ext) if prev_path and prev_path.exists() else cur
    nxt = read_frame_uint8(next_path, image_ext) if next_path and next_path.exists() else cur
    rgb = np.stack([prev, cur, nxt], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    img = resize_pil(img, size_hw)
    return np.array(img, dtype=np.uint8)


def read_mask(
    mask_path: Path,
    resize: Optional[Tuple[int, int]],
    fallback_size: Tuple[int, int],
):
    img = Image.open(mask_path).convert("L")
    if resize:
        img = resize_pil(img, resize)
    elif img.size != fallback_size:
        img = img.resize(fallback_size, resample=Image.BILINEAR)
    arr = np.array(img)
    return arr > 127


def sequences_from_dict(datadir: Path, sequences_dict: dict) -> list[tuple[str, str]]:
    try:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from helpers import sequences_from_dict as helper_sequences_from_dict

        return helper_sequences_from_dict(str(datadir), sequences_dict)
    except Exception:
        sequences = []
        for subj, seqs in sequences_dict.items():
            use_seqs = seqs
            if len(seqs) == 0:
                subj_dir = datadir / subj
                use_seqs = [
                    s for s in os.listdir(subj_dir) if (subj_dir / s).is_dir()
                ]
            sequences.extend([(subj, seq) for seq in use_seqs])
        return sequences


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read config files.")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config file.")
    return cfg


def load_exclusion_list(path: Optional[str]):
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {tuple(item) for item in data}


def build_label_map(
    classes_sorted,
    mask_dir: Path,
    image_name: str,
    size_hw: Optional[Tuple[int, int]],
    fallback_size: Tuple[int, int],
):
    if size_hw:
        label_map = np.zeros((size_hw[0], size_hw[1]), dtype=np.uint16)
    else:
        label_map = np.zeros((fallback_size[1], fallback_size[0]), dtype=np.uint16)
    for idx, art in enumerate(classes_sorted):
        mask_path = mask_dir / f"{image_name}_{art}.png"
        if not mask_path.exists():
            continue
        mask_bin = read_mask(mask_path, size_hw, fallback_size)
        label_map[mask_bin] = idx + 1
    return label_map


def process_original_split(
    split: str,
    cfg: dict,
    output_root: Path,
    include_empty: bool,
    allow_missing: bool,
    resize: Optional[Tuple[int, int]],
    image_folder_override: Optional[str],
    image_ext_override: Optional[str],
    max_frames_per_seq: Optional[int],
    exclusion_list: set,
):
    datadir = Path(os.path.expanduser(cfg["datadir"]))
    image_folder = image_folder_override or cfg.get("image_folder", "dicoms")
    image_ext = (image_ext_override or cfg.get("image_ext", "dcm")).lower()
    classes_sorted = sorted(cfg["classes"])
    mode = cfg.get("mode", "gray").lower()
    size_hw = resize if resize else cfg.get("size")
    if size_hw:
        size_hw = (int(size_hw[0]), int(size_hw[1]))

    split_key = "valid" if split in ("val", "valid") else split
    sequences_dict = cfg.get(f"{split_key}_sequences", {})

    if not sequences_dict:
        print(f"[skip] no sequences found for split '{split}' in config")
        return

    sequences = sequences_from_dict(datadir, sequences_dict)
    out_split_dir = output_root / split

    for subject, sequence in sequences:
        seq_dir = datadir / subject / sequence
        img_dir = seq_dir / image_folder
        mask_dir = seq_dir / "masks"
        if not img_dir.exists():
            print(f"[skip] missing image folder: {img_dir}")
            continue

        image_paths = sorted(
            img_dir.glob(f"*.{image_ext}"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if not image_paths:
            print(f"[skip] no images in {img_dir}")
            continue

        imgs = []
        gts = []
        kept = 0

        for img_path in image_paths:
            image_name = img_path.stem
            if image_name.isdigit():
                instance_number = int(image_name)
                if (subject, sequence, instance_number) in exclusion_list:
                    continue
            mask_paths = {
                art: mask_dir / f"{image_name}_{art}.png" for art in classes_sorted
            }
            missing = [art for art, p in mask_paths.items() if not p.exists()]
            if missing and not allow_missing:
                continue
            if len(missing) == len(classes_sorted) and not include_empty:
                continue

            if mode == "rgb":
                img_arr = load_rgb_triplet(img_path, image_ext, size_hw)
                height, width = img_arr.shape[:2]
            else:
                frame = read_frame_uint8(img_path, image_ext)
                img = Image.fromarray(frame, mode="L")
                img = resize_pil(img, size_hw)
                img_arr = np.array(img, dtype=np.uint8)
                height, width = img_arr.shape[:2]

            label_map = build_label_map(
                classes_sorted,
                mask_dir,
                image_name,
                size_hw,
                (width, height),
            )

            imgs.append(img_arr)
            gts.append(label_map)
            kept += 1
            if max_frames_per_seq and kept >= max_frames_per_seq:
                break

        if not imgs:
            continue

        imgs_arr = np.stack(imgs, axis=0)
        gts_arr = np.stack(gts, axis=0)

        out_path = out_split_dir / subject / f"{sequence}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, imgs=imgs_arr, gts=gts_arr)

    print(f"[done] {split}: wrote NPZ to {out_split_dir}")


def visualize_npz_samples(npz_root: Path, count: int, out_dir: Path, seed: int):
    npz_files = sorted(npz_root.rglob("*.npz"))
    if not npz_files:
        print(f"[skip] no npz files found in {npz_root}")
        return
    rng = random.Random(seed)
    if count < len(npz_files):
        npz_files = rng.sample(npz_files, count)

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
    ]

    for npz_path in npz_files:
        data = np.load(npz_path)
        imgs = data["imgs"]
        gts = data["gts"]
        frame_idx = 0

        if imgs.ndim == 3:
            img = imgs[frame_idx]
            img_rgb = np.stack([img, img, img], axis=-1)
        elif imgs.ndim == 4 and imgs.shape[-1] == 3:
            img_rgb = imgs[frame_idx]
        elif imgs.ndim == 4 and imgs.shape[1] == 3:
            img_rgb = imgs[frame_idx].transpose(1, 2, 0)
        else:
            print(f"[skip] unsupported imgs shape in {npz_path}: {imgs.shape}")
            continue

        mask = gts[frame_idx]
        color_mask = np.zeros_like(img_rgb)
        for obj_id in np.unique(mask):
            if obj_id == 0:
                continue
            color = palette[(obj_id - 1) % len(palette)]
            color_mask[mask == obj_id] = color

        overlay = (0.6 * img_rgb + 0.4 * color_mask).astype(np.uint8)
        left = Image.fromarray(img_rgb.astype(np.uint8))
        right = Image.fromarray(overlay)
        canvas = Image.new("RGB", (left.width + right.width, left.height))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))

        rel = npz_path.relative_to(npz_root)
        out_path = out_dir / f"{rel.as_posix().replace('/', '__')}_f{frame_idx}.png"
        canvas.save(out_path)
    print(f"[done] saved visualizations to {out_dir}")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.source == "yolo":
        yolo_root = Path(args.yolo_root)
        for split in splits:
            process_yolo_split(
                split=split,
                yolo_root=yolo_root,
                output_root=output_root,
                group_by=args.group_by,
                include_empty=args.include_empty,
            )
    else:
        if not args.config:
            raise SystemExit("--config is required for source=original")
        cfg = load_config(Path(args.config))
        resize = tuple(args.resize) if args.resize else None
        exclusion_list = load_exclusion_list(args.exclusion_list)
        for split in splits:
            process_original_split(
                split=split,
                cfg=cfg,
                output_root=output_root,
                include_empty=args.include_empty,
                allow_missing=args.allow_missing,
                resize=resize,
                image_folder_override=args.image_folder,
                image_ext_override=args.image_ext,
                max_frames_per_seq=args.max_frames_per_seq,
                exclusion_list=exclusion_list,
            )

    if args.visualize:
        viz_root = output_root / args.viz_split
        visualize_npz_samples(
            npz_root=viz_root,
            count=args.viz_count,
            out_dir=Path(args.viz_out),
            seed=args.viz_seed,
        )

if __name__ == "__main__":
    main()
