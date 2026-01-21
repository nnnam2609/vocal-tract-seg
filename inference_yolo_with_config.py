"""
YOLO Segmentation Inference and Evaluation Script with Config Support
Performs inference on test images, calculates metrics (P2CP, Jaccard), and visualizes results
Supports both command-line arguments and YAML config files
Uses vt_tools and vt_tracker for contour processing (same as MaskRCNN)
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLO
import yaml
from tqdm import tqdm
from datetime import datetime
import time

# Import vt_tools and vt_tracker (same as MaskRCNN)
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker.postprocessing import POST_PROCESSING
from vt_tracker.postprocessing.calculate_contours import calculate_contour


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_image_name(image_name):
    """
    Parse image filename to extract subject, sequence, and frame
    Handles two formats:
    1. ArtSpeech_Database_2_{subject}_{sequence}_{frame}.jpg
       Example: ArtSpeech_Database_2_1775_S9_1057.jpg -> subject=1775, sequence=S9, frame=1057
    2. ArtSpeech_Vocal_Tract_Segmentation_{subject}_{sequence}_{frame}.jpg
       Example: ArtSpeech_Vocal_Tract_Segmentation_1662_S10_1831.jpg -> subject=1662, sequence=S10, frame=1831
    """
    # Remove extension
    base_name = os.path.splitext(image_name)[0]
    
    # Split by underscore
    parts = base_name.split('_')
    
    # Default values
    subject = None
    sequence = None
    frame = None
    
    try:
        # Format 1: ArtSpeech_Database_2_{subject}_{sequence}_{frame}
        # Parts: [ArtSpeech, Database, 2, subject, sequence, frame]
        if len(parts) >= 6 and parts[1] == 'Database':
            subject = parts[3]  # 1775
            sequence = parts[4]  # S9
            frame = int(parts[5])  # 1057
        
        # Format 2: ArtSpeech_Vocal_Tract_Segmentation_{subject}_{sequence}_{frame}
        # Parts: [ArtSpeech, Vocal, Tract, Segmentation, subject, sequence, frame]
        elif len(parts) >= 7 and parts[1] == 'Vocal' and parts[2] == 'Tract':
            subject = parts[4]  # 1662
            sequence = parts[5]  # S10
            frame = int(parts[6])  # 1831
        
        # Fallback: try to find pattern {number}_{S##}_{number} at the end
        else:
            # Find the last 3 parts that match the pattern
            for i in range(len(parts) - 2):
                try:
                    if parts[i+1].startswith('S') and parts[i].isdigit() and parts[i+2].isdigit():
                        subject = parts[i]
                        sequence = parts[i+1]
                        frame = int(parts[i+2])
                        break
                except:
                    continue
    except Exception as e:
        print(f"Warning: Could not parse image name '{image_name}': {e}")
    
    return subject, sequence, frame


def smooth_contour(contour):
    """Smooth contour using B-spline regularization (from vt_tools - same as MaskRCNN)"""
    try:
        resX, resY = regularize_Bsplines(contour, 3)
        return np.array([resX, resY]).T
    except:
        return contour


def load_articulator_array(filepath):
    """
    Loads the target array with the proper orientation (right to left)
    (same as MaskRCNN inference)
    """
    target_array = np.load(filepath)

    # All the contours should be oriented from right to left. If it is the opposite,
    # we flip the array.
    if target_array[0][0] < target_array[-1][0]:
        target_array = np.flip(target_array, axis=0)

    return target_array.copy()


def extract_contour_from_mask(binary_mask, class_name=None, use_vt_tracker=True, gravity_curve=None):
    """
    Extract contour from binary mask
    If use_vt_tracker=True, uses vt_tracker.calculate_contour (same as MaskRCNN)
    Otherwise uses simple OpenCV contour extraction
    """
    if use_vt_tracker and class_name is not None:
        # Use vt_tracker post-processing (same as MaskRCNN) - SLOW but accurate
        try:
            # Normalize mask to 0-1 range
            mask_normalized = binary_mask.astype(np.float32) / 255.0
            
            # Get post-processing config for this class
            post_proc_cfg = deepcopy(POST_PROCESSING.get(class_name, {}))
            
            # Calculate contour using vt_tracker
            contour = calculate_contour(
                class_name, 
                mask_normalized, 
                gravity_curve=gravity_curve, 
                cfg=post_proc_cfg
            )
            
            if len(contour) > 0:
                # Smooth the contour
                contour = smooth_contour(contour)
                return contour
            else:
                return None
        except Exception as e:
            print(f"    Warning: vt_tracker failed for {class_name}, falling back to OpenCV: {e}")
            # Fall back to OpenCV if vt_tracker fails
            pass
    
    # Simple OpenCV extraction (fallback)
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE 
    )
    if len(contours) == 0:
        return None
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()
    
    if len(contour.shape) == 1:  # Single point
        return None
    
    # Convert to (N, 2) array of (x, y) coordinates
    contour_array = np.array([[pt[0], pt[1]] for pt in contour])
    
    # Orient from right to left
    if contour_array[0][0] < contour_array[-1][0]:
        contour_array = np.flip(contour_array, axis=0)
    
    return contour_array


def regularize_contour_bspline(contour, degree=2, num_points=100):
    """Regularize contour using B-spline interpolation (backup method)"""
    if contour is None or len(contour) < 4:
        return None
    
    # Just use smooth_contour from vt_tools
    return smooth_contour(contour)


def point_to_curve_distance(point, curve):
    """Calculate minimum distance from a point to a curve"""
    distances = np.sqrt(np.sum((curve - point)**2, axis=1))
    return np.min(distances)


def p2cp_mean_distance(pred_contour, gt_contour):
    """Calculate mean point-to-curve-projection distance"""
    distances = [point_to_curve_distance(pt, gt_contour) for pt in pred_contour]
    return np.mean(distances)


def p2cp_rms_distance(pred_contour, gt_contour):
    """Calculate RMS point-to-curve-projection distance"""
    distances = [point_to_curve_distance(pt, gt_contour) for pt in pred_contour]
    return np.sqrt(np.mean(np.array(distances)**2))


def jaccard_index(pred_mask, gt_mask, eps=1e-15):
    """Calculate Jaccard Index (IoU)"""
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return (intersection + eps) / (union - intersection + eps)


def create_filled_mask_from_contour(contour, shape):
    """Create a filled binary mask from contour"""
    mask = np.zeros(shape, dtype=np.uint8)
    if contour is None or len(contour) < 3:
        return mask
    
    contour_int = contour.astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 1)
    mask = binary_fill_holes(mask).astype(int)
    return mask


def load_ground_truth_mask(label_path, img_shape):
    """
    Load YOLO format label and convert to segmentation mask
    YOLO format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    """
    if not os.path.exists(label_path):
        return None
    
    h, w = img_shape[:2]
    masks = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            
            # Denormalize coordinates
            coords[:, 0] *= w
            coords[:, 1] *= h
            coords = coords.astype(np.int32)
            
            # Create mask for this class
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 255)
            masks[class_id] = mask
    
    return masks


def evaluate_single_image(model, image_path, label_path, config):
    """
    Perform inference and evaluation on a single image
    
    Args:
        model: YOLO model
        image_path: Path to test image
        label_path: Path to ground truth label (YOLO format)
        config: Configuration dictionary
    
    Returns:
        dict: Evaluation metrics for each class
    """
    class_names = config['labels']
    closed_articulators = config['evaluation']['closed_articulators']
    inference_cfg = config['inference']
    postproc_cfg = config['postprocessing']
    
    timings = {}
    total_start = time.time()
    
    # Load image
    t0 = time.time()
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️  Could not load image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    timings['load_image'] = time.time() - t0
    
    # Run inference
    t0 = time.time()
    results = model.predict(
        image_path, 
        conf=inference_cfg.get('conf_threshold', 0.25),
        iou=inference_cfg.get('iou_threshold', 0.65),
        max_det=inference_cfg.get('max_det', 300),
        imgsz=inference_cfg.get('imgsz', 640),
        augment=inference_cfg.get('augment', False),
        half=inference_cfg.get('half', False),
        verbose=False
    )[0]
    timings['inference'] = time.time() - t0
    
    # Load ground truth
    t0 = time.time()
    gt_masks = load_ground_truth_mask(label_path, img.shape)
    
    if gt_masks is None:
        print(f"  ⚠️  No ground truth found at: {label_path}")
        return None
    timings['load_gt'] = time.time() - t0
    
    # Extract predictions
    t0 = time.time()
    pred_masks = {}
    if results.masks is not None:
        for i, (box, mask, cls) in enumerate(zip(results.boxes, results.masks, results.boxes.cls)):
            class_id = int(cls.item())
            conf = box.conf.item()
            
            # Get mask as numpy array
            mask_array = mask.data[0].cpu().numpy()
            
            # Resize mask to original image size
            mask_resized = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Store prediction (keep highest confidence if multiple)
            if class_id not in pred_masks or conf > pred_masks[class_id]['conf']:
                pred_masks[class_id] = {
                    'mask': mask_binary,
                    'conf': conf
                }
    
    timings['extract_masks'] = time.time() - t0
    
    # Calculate metrics for each class
    t0 = time.time()
    evaluation_results = []
    
    # Parse image name to extract subject, sequence, frame
    image_name = os.path.basename(image_path)
    subject, sequence, frame = parse_image_name(image_name)
    
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        
        result = {
            'image_name': image_name,
            'subject': subject,
            'sequence': sequence,
            'frame': frame,
            'class_id': class_id,
            'class_name': class_name,
            'p2cp_mean': np.nan,
            'p2cp_rms': np.nan,
            'jaccard_index': np.nan,
            'has_prediction': class_id in pred_masks,
            'has_ground_truth': class_id in gt_masks,
            'confidence': pred_masks[class_id]['conf'] if class_id in pred_masks else 0.0,
            'pred_pixels': 0,
            'gt_pixels': 0
        }
        
        # Skip if no ground truth for this class
        if class_id not in gt_masks:
            evaluation_results.append(result)
            continue
        
        gt_mask = gt_masks[class_id]
        result['gt_pixels'] = int((gt_mask > 0).sum())
        result['has_ground_truth'] = True
        
        # Skip if no prediction for this class
        if class_id not in pred_masks:
            evaluation_results.append(result)
            continue
        
        pred_mask = pred_masks[class_id]['mask']
        result['pred_pixels'] = int((pred_mask > 0).sum())
        result['has_prediction'] = True
        
        try:
            # Determine if we should use vt_tracker post-processing
            use_vt_tracker = postproc_cfg.get('use_vt_tracker', True)
            
            # Get gravity curve for this image/class if available (for upper-incisor dependent articulators)
            gravity_curve = None
            
            # Extract contours using vt_tracker (same as MaskRCNN)
            pred_contour = extract_contour_from_mask(
                pred_mask, 
                class_name=class_name if use_vt_tracker else None,
                use_vt_tracker=use_vt_tracker,
                gravity_curve=gravity_curve
            )
            
            gt_contour = extract_contour_from_mask(
                gt_mask,
                class_name=class_name if use_vt_tracker else None,
                use_vt_tracker=use_vt_tracker,
                gravity_curve=gravity_curve
            )
            
            if pred_contour is None or gt_contour is None:
                evaluation_results.append(result)
                continue
            
            # Contours are already smoothed if use_vt_tracker=True
            # No need for additional regularization
            reg_pred = pred_contour
            reg_gt = gt_contour
            
            # Calculate P2CP metrics
            result['p2cp_mean'] = p2cp_mean_distance(reg_pred, reg_gt)
            result['p2cp_rms'] = p2cp_rms_distance(reg_pred, reg_gt)
            
            # Calculate Jaccard for closed articulators
            if class_name in closed_articulators:
                pred_filled = create_filled_mask_from_contour(pred_contour, img.shape[:2])
                gt_filled = create_filled_mask_from_contour(gt_contour, img.shape[:2])
                result['jaccard_index'] = jaccard_index(pred_filled, gt_filled)
            
        except Exception as e:
            print(f"    Error processing {class_name}: {e}")
        
        evaluation_results.append(result)
    
    timings['calculate_metrics'] = time.time() - t0
    timings['total'] = time.time() - total_start
    
    # Print timing info if verbose
    if config.get('inference', {}).get('verbose', False):
        print(f"  ⏱️  Timing for {os.path.basename(image_path)}:")
        print(f"    Load image: {timings['load_image']*1000:.1f}ms")
        print(f"    Inference:  {timings['inference']*1000:.1f}ms")
        print(f"    Load GT:    {timings['load_gt']*1000:.1f}ms")
        print(f"    Extract masks: {timings['extract_masks']*1000:.1f}ms")
        print(f"    Calculate metrics: {timings['calculate_metrics']*1000:.1f}ms")
        print(f"    Total:      {timings['total']*1000:.1f}ms")
    
    return {
        'results': evaluation_results,
        'img': img_rgb,
        'pred_masks': pred_masks,
        'gt_masks': gt_masks,
        'timings': timings
    }


def visualize_results(img, pred_masks, gt_masks, results, output_path, config):
    """Create comprehensive visualization of results with raw segmentation and final contours"""
    
    viz_start = time.time()
    
    class_names = config['labels']
    viz_cfg = config.get('visualization', {})
    h, w = img.shape[:2]
    
    # Get line styles from config
    gt_linestyle = viz_cfg.get('gt_linestyle', '-')  # Solid for GT
    pred_linestyle = viz_cfg.get('pred_linestyle', '--')  # Dashed for prediction
    show_raw = viz_cfg.get('show_raw_segmentation', True)
    use_vt_tracker = config.get('postprocessing', {}).get('use_vt_tracker', True)
    verbose = config.get('inference', {}).get('verbose', False)
    
    # Pre-extract all contours once (optimization - avoid redundant extractions)
    gt_contours = {}
    pred_contours = {}
    
    t_contour = time.time()
    for class_id, mask in gt_masks.items():
        class_name = class_names[class_id]
        contour = extract_contour_from_mask(
            mask,
            class_name=class_name if use_vt_tracker else None,
            use_vt_tracker=use_vt_tracker
        )
        if contour is not None:
            gt_contours[class_id] = contour
    
    for class_id, pred_data in pred_masks.items():
        class_name = class_names[class_id]
        mask = pred_data['mask']
        contour = extract_contour_from_mask(
            mask,
            class_name=class_name if use_vt_tracker else None,
            use_vt_tracker=use_vt_tracker
        )
        if contour is not None:
            pred_contours[class_id] = contour
    
    contour_time = time.time() - t_contour
    
    # Create figure with multiple subplots (3x3 grid)
    t_plot = time.time()
    fig, axes = plt.subplots(3, 3, figsize=viz_cfg.get('figsize', [20, 14]))
    fig.suptitle(f'YOLO Segmentation Evaluation - {os.path.basename(output_path)}', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Original overlays
    # 1. Original Image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Ground Truth Overlay
    gt_composite = np.zeros((h, w), dtype=np.uint8)
    for class_id, mask in gt_masks.items():
        gt_composite[mask > 0] = class_id + 1
    
    axes[0, 1].imshow(img, alpha=0.6)
    axes[0, 1].imshow(gt_composite, cmap='tab10', 
                      alpha=viz_cfg.get('overlay_alpha', 0.5), vmin=0, vmax=10)
    axes[0, 1].set_title('Ground Truth Overlay', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Prediction Overlay
    pred_composite = np.zeros((h, w), dtype=np.uint8)
    for class_id, pred_data in pred_masks.items():
        mask = pred_data['mask']
        pred_composite[mask > 0] = class_id + 1
    
    axes[0, 2].imshow(img, alpha=0.6)
    axes[0, 2].imshow(pred_composite, cmap='tab10', 
                      alpha=viz_cfg.get('overlay_alpha', 0.5), vmin=0, vmax=10)
    axes[0, 2].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Raw segmentation masks
    use_vt_tracker = config.get('postprocessing', {}).get('use_vt_tracker', True)
    
    if show_raw:
        # 4. GT Raw Segmentation
        axes[1, 0].imshow(gt_composite, cmap='tab10', vmin=0, vmax=10)
        axes[1, 0].set_title('GT Raw Segmentation', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 5. Prediction Raw Segmentation
        axes[1, 1].imshow(pred_composite, cmap='tab10', vmin=0, vmax=10)
        axes[1, 1].set_title('Prediction Raw Segmentation', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # 6. Comparison (both raw masks)
        # Create RGB comparison
        comparison_img = np.zeros((h, w, 3), dtype=np.uint8)
        # GT in red channel
        comparison_img[:, :, 0] = (gt_composite > 0).astype(np.uint8) * 255
        # Prediction in green channel
        comparison_img[:, :, 1] = (pred_composite > 0).astype(np.uint8) * 255
        # Overlap will be yellow (red + green)
        
        axes[1, 2].imshow(img, alpha=0.5)
        axes[1, 2].imshow(comparison_img, alpha=0.6)
        axes[1, 2].set_title('Raw Mask Comparison\n(GT=Red, Pred=Green, Overlap=Yellow)', 
                           fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')
    else:
        # Hide row 2 if not showing raw segmentation
        for ax in axes[1, :]:
            ax.axis('off')
    
    # Row 3: Final contours (use pre-extracted contours)
    # 7. GT Contours (solid lines)
    axes[2, 0].imshow(img, alpha=0.5)
    for class_id, contour in gt_contours.items():
        class_name = class_names[class_id]
        axes[2, 0].plot(contour[:, 0], contour[:, 1], 
                       linewidth=viz_cfg.get('contour_linewidth', 2.5),
                       linestyle=gt_linestyle,
                       label=class_name)
    axes[2, 0].set_title('GT Final Contours (Solid)', fontsize=12, fontweight='bold')
    axes[2, 0].legend(loc='upper right', fontsize=7)
    axes[2, 0].axis('off')
    
    # 8. Predicted Contours (dashed lines)
    axes[2, 1].imshow(img, alpha=0.5)
    show_conf = viz_cfg.get('show_confidence', True)
    for class_id, contour in pred_contours.items():
        class_name = class_names[class_id]
        conf = pred_masks[class_id]['conf']
        label = f"{class_name} ({conf:.2f})" if show_conf else class_name
        axes[2, 1].plot(contour[:, 0], contour[:, 1],
                       linewidth=viz_cfg.get('contour_linewidth', 2.5),
                       linestyle=pred_linestyle,
                       label=label)
    axes[2, 1].set_title('Prediction Final Contours (Dashed)', fontsize=12, fontweight='bold')
    axes[2, 1].legend(loc='upper right', fontsize=7)
    axes[2, 1].axis('off')
    
    # 9. GT vs Prediction Contour Comparison
    axes[2, 2].imshow(img, alpha=0.5)
    
    # Draw GT contours (solid)
    for class_id, contour in gt_contours.items():
        class_name = class_names[class_id]
        axes[2, 2].plot(contour[:, 0], contour[:, 1], 
                       linewidth=2, linestyle=gt_linestyle, alpha=0.8,
                       label=f"GT: {class_name}")
    
    # Draw prediction contours (dashed)
    for class_id, contour in pred_contours.items():
        class_name = class_names[class_id]
        axes[2, 2].plot(contour[:, 0], contour[:, 1],
                       linewidth=2.5, linestyle=pred_linestyle, alpha=0.9,
                       label=f"Pred: {class_name}")
    
    axes[2, 2].set_title('GT vs Pred Contour Overlay\n(GT=Solid, Pred=Dashed)', 
                        fontsize=11, fontweight='bold')
    axes[2, 2].legend(loc='upper right', fontsize=6)
    axes[2, 2].axis('off')
    
    plot_time = time.time() - t_plot
    
    plt.tight_layout()
    
    # Save figure
    t_save = time.time()
    plt.savefig(output_path, dpi=viz_cfg.get('dpi', 200), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    save_time = time.time() - t_save
    
    total_viz_time = time.time() - viz_start
    
    # Only print detailed timing if verbose (don't interrupt tqdm progress bar)
    if verbose:
        print(f"\n   Visualization timing for {os.path.basename(output_path)}:")
        print(f"     Contour extraction: {contour_time*1000:.1f}ms")
        print(f"     Plotting:           {plot_time*1000:.1f}ms")
        print(f"     Saving:             {save_time*1000:.1f}ms")
        print(f"     Total:              {total_viz_time*1000:.1f}ms")
    
    return total_viz_time


def batch_evaluate(model, config):
    """Evaluate multiple images from test set with batch processing support"""
    
    batch_start_time = time.time()
    
    input_folder = config['paths']['input_folder']
    output_folder = config['paths']['output_folder']
    gt_folder = config['paths']['ground_truth_folder']
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of images to process
    if config['data']['process_all']:
        image_files = sorted([f for f in os.listdir(input_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    else:
        image_files = config['data'].get('specific_images', [])
    
    if len(image_files) == 0:
        print(" No images found to process!")
        return None
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"GT folder:     {gt_folder}\n")
    
    inference_cfg = config['inference']
    use_batch = inference_cfg.get('use_batch_predict', False)
    batch_size = inference_cfg.get('batch_size', 1)
    
    all_results = []
    total_inference_time = 0.0
    total_processing_time = 0.0
    total_viz_time = 0.0
    
    if use_batch and batch_size > 1:
        print(f"Using batch inference with batch_size={batch_size}")
        
        # Process in batches
        for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[batch_start:batch_start + batch_size]
            batch_paths = [os.path.join(input_folder, f) for f in batch_files]
            
            # Run batch inference ONCE for all images in this batch (optimal for GPU)
            try:
                t_inf = time.time()
                results_batch = model.predict(
                    batch_paths,
                    conf=inference_cfg.get('conf_threshold', 0.25),
                    iou=inference_cfg.get('iou_threshold', 0.65),
                    max_det=inference_cfg.get('max_det', 300),
                    imgsz=inference_cfg.get('imgsz', 640),
                    augment=inference_cfg.get('augment', False),
                    half=inference_cfg.get('half', False),
                    verbose=True,
                    device=inference_cfg.get('device', 0)
                )
                total_inference_time += time.time() - t_inf
                
                # Process each result in the batch
                for img_file, yolo_result in zip(batch_files, results_batch):
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(gt_folder, label_file)
                    image_path = os.path.join(input_folder, img_file)
                    
                    if not os.path.exists(label_path):
                        print(f" No label found for {img_file}, skipping...")
                        continue
                    
                    try:
                        t_proc = time.time()
                        eval_data = process_single_result(
                            yolo_result, image_path, label_path, config
                        )
                        total_processing_time += time.time() - t_proc
                        
                        if eval_data is None:
                            continue
                        
                        results = eval_data['results']
                        all_results.extend(results)
                        
                        # Save visualization if enabled
                        if config['evaluation'].get('save_visualizations', True):
                            if config['evaluation'].get('save_per_image', True):
                                viz_path = os.path.join(output_folder, f'eval_{os.path.splitext(img_file)[0]}.png')
                                viz_time = visualize_results(
                                    eval_data['img'],
                                    eval_data['pred_masks'],
                                    eval_data['gt_masks'],
                                    results,
                                    viz_path,
                                    config
                                )
                                total_viz_time += viz_time
                    
                    except Exception as e:
                        print(f"Error processing {img_file}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f" Error in batch: {str(e)}")
                continue
    else:
        # Single image processing (original)
        print("Using single-image inference")
        for img_file in tqdm(image_files, desc="Evaluating images"):
            image_path = os.path.join(input_folder, img_file)
            
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(gt_folder, label_file)
            
            if not os.path.exists(label_path):
                print(f"  ⚠️  No label found for {img_file}, skipping...")
                continue
            
            try:
                t_proc = time.time()
                eval_data = evaluate_single_image(model, image_path, label_path, config)
                
                if eval_data is None:
                    continue
                
                # Track timing
                if 'timings' in eval_data:
                    total_inference_time += eval_data['timings'].get('inference', 0)
                    total_processing_time += eval_data['timings'].get('total', 0) - eval_data['timings'].get('inference', 0)
                
                results = eval_data['results']
                all_results.extend(results)
                
                # Save visualization if enabled
                if config['evaluation'].get('save_visualizations', True):
                    if config['evaluation'].get('save_per_image', True):
                        viz_path = os.path.join(output_folder, f'eval_{os.path.splitext(img_file)[0]}.png')
                        viz_time = visualize_results(
                            eval_data['img'],
                            eval_data['pred_masks'],
                            eval_data['gt_masks'],
                            results,
                            viz_path,
                            config
                        )
                        total_viz_time += viz_time
            
            except Exception as e:
                print(f"  ✗ Error processing {img_file}: {str(e)}")
                continue
    
    # Print timing summary
    total_time = time.time() - batch_start_time
    n_images = len(all_results) // len(config['labels']) if len(all_results) > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"⏱️  TIMING SUMMARY - {n_images} images processed")
    print(f"{'='*80}")
    print(f"  Total time:        {total_time:.2f}s ({total_time/60:.1f}m)")
    if n_images > 0:
        print(f"  Per image:         {total_time/n_images:.2f}s")
        print(f"  ")
        other_time = total_time - total_inference_time - total_processing_time - total_viz_time
        print(f"Inference:       {total_inference_time:.2f}s ({total_inference_time/n_images:.3f}s/img) - {total_inference_time/total_time*100:.1f}%")
        print(f"Processing:      {total_processing_time:.2f}s ({total_processing_time/n_images:.3f}s/img) - {total_processing_time/total_time*100:.1f}%")
        print(f"Visualization:   {total_viz_time:.2f}s ({total_viz_time/n_images:.3f}s/img) - {total_viz_time/total_time*100:.1f}%")
        if other_time > 0:
            print(f"Other overhead:  {other_time:.2f}s - {other_time/total_time*100:.1f}%")
    print(f"{'='*80}\n")
    
    return all_results


def process_single_result(yolo_result, image_path, label_path, config):
    """
    Process a single YOLO result (used in batch processing)
    This function does NOT run inference - it receives pre-computed results
    from batch inference and only processes metrics/visualization.
    
    Args:
        yolo_result: Pre-computed YOLO result from batch inference
        image_path: Path to the image file
        label_path: Path to ground truth label
        config: Configuration dictionary
    """
    class_names = config['labels']
    closed_articulators = config['evaluation']['closed_articulators']
    postproc_cfg = config['postprocessing']
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️  Could not load image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Load ground truth
    gt_masks = load_ground_truth_mask(label_path, img.shape)
    
    if gt_masks is None:
        print(f"  ⚠️  No ground truth found at: {label_path}")
        return None
    
    # Extract predictions from YOLO result
    pred_masks = {}
    if yolo_result.masks is not None:
        for i, (box, mask, cls) in enumerate(zip(yolo_result.boxes, yolo_result.masks, yolo_result.boxes.cls)):
            class_id = int(cls.item())
            conf = box.conf.item()
            
            # Get mask as numpy array
            mask_array = mask.data[0].cpu().numpy()
            
            # Resize mask to original image size
            mask_resized = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Store prediction (keep highest confidence if multiple)
            if class_id not in pred_masks or conf > pred_masks[class_id]['conf']:
                pred_masks[class_id] = {
                    'mask': mask_binary,
                    'conf': conf
                }
    
    # Calculate metrics for each class
    evaluation_results = []
    
    # Parse image name to extract subject, sequence, frame
    image_name = os.path.basename(image_path)
    subject, sequence, frame = parse_image_name(image_name)
    
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        
        result = {
            'image_name': image_name,
            'subject': subject,
            'sequence': sequence,
            'frame': frame,
            'class_id': class_id,
            'class_name': class_name,
            'p2cp_mean': np.nan,
            'p2cp_rms': np.nan,
            'jaccard_index': np.nan,
            'has_prediction': class_id in pred_masks,
            'has_ground_truth': class_id in gt_masks,
            'confidence': pred_masks[class_id]['conf'] if class_id in pred_masks else 0.0,
            'pred_pixels': 0,
            'gt_pixels': 0
        }
        
        # Skip if no ground truth for this class
        if class_id not in gt_masks:
            evaluation_results.append(result)
            continue
        
        gt_mask = gt_masks[class_id]
        result['gt_pixels'] = int((gt_mask > 0).sum())
        result['has_ground_truth'] = True
        
        # Skip if no prediction for this class
        if class_id not in pred_masks:
            evaluation_results.append(result)
            continue
        
        pred_mask = pred_masks[class_id]['mask']
        result['pred_pixels'] = int((pred_mask > 0).sum())
        result['has_prediction'] = True
        
        try:
            # Determine if we should use vt_tracker post-processing
            use_vt_tracker = postproc_cfg.get('use_vt_tracker', True)
            
            # Extract contours using vt_tracker (same as MaskRCNN)
            pred_contour = extract_contour_from_mask(
                pred_mask, 
                class_name=class_name if use_vt_tracker else None,
                use_vt_tracker=use_vt_tracker,
                gravity_curve=None
            )
            
            gt_contour = extract_contour_from_mask(
                gt_mask,
                class_name=class_name if use_vt_tracker else None,
                use_vt_tracker=use_vt_tracker,
                gravity_curve=None
            )
            
            if pred_contour is None or gt_contour is None:
                evaluation_results.append(result)
                continue
            
            reg_pred = pred_contour
            reg_gt = gt_contour
            
            # Calculate P2CP metrics
            result['p2cp_mean'] = p2cp_mean_distance(reg_pred, reg_gt)
            result['p2cp_rms'] = p2cp_rms_distance(reg_pred, reg_gt)
            
            # Calculate Jaccard for closed articulators
            if class_name in closed_articulators:
                pred_filled = create_filled_mask_from_contour(pred_contour, img.shape[:2])
                gt_filled = create_filled_mask_from_contour(gt_contour, img.shape[:2])
                result['jaccard_index'] = jaccard_index(pred_filled, gt_filled)
            
        except Exception as e:
            print(f"    Error processing {class_name}: {e}")
        
        evaluation_results.append(result)
    
    return {
        'results': evaluation_results,
        'img': img_rgb,
        'pred_masks': pred_masks,
        'gt_masks': gt_masks
    }


def save_results(results, config):
    """Save evaluation results to CSV and summary files"""
    
    output_folder = config['paths']['output_folder']
    
    if not results or len(results) == 0:
        print("No results to save!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to have subject, sequence, frame at the beginning
    column_order = ['subject', 'sequence', 'frame', 'image_name', 'class_id', 'class_name', 
                    'jaccard_index', 'has_prediction', 'has_ground_truth', 
                    'confidence', 'pred_pixels', 'gt_pixels']
    
    # Only include columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    
    # Add any remaining columns that weren't in our predefined order
    remaining_cols = [col for col in df.columns if col not in column_order]
    column_order.extend(remaining_cols)
    
    df = df[column_order]
    
    # Save detailed results
    if config['evaluation'].get('save_csv', True):
        csv_path = os.path.join(output_folder, 'evaluation_results_detailed.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # Generate and save summary
    if config['evaluation'].get('save_summary', True):
        valid_results = df[df['has_prediction'] & df['has_ground_truth']]
        
        summary_path = os.path.join(output_folder, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("YOLO SEGMENTATION EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {config.get('_config_path', 'N/A')}\n")
            f.write(f"Model: {config['model']['weights']}\n\n")
            
            f.write(f"Total images processed: {df['image_name'].nunique()}\n")
            f.write(f"Total class predictions: {len(valid_results)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            
            for class_id in sorted(df['class_id'].unique()):
                class_data = valid_results[valid_results['class_id'] == class_id]
                
                if len(class_data) > 0:
                    class_name = class_data.iloc[0]['class_name']
                    f.write(f"{class_name.upper()}\n")
                    f.write(f"  Images with predictions: {len(class_data)}\n")
                    
                    p2cp_means = class_data['p2cp_mean'].dropna()
                    p2cp_rms = class_data['p2cp_rms'].dropna()
                    jaccards = class_data['jaccard_index'].dropna()
                    
                    if len(p2cp_means) > 0:
                        f.write(f"  P2CP Mean: {p2cp_means.mean():.4f} ± {p2cp_means.std():.4f} px\n")
                        f.write(f"  P2CP RMS:  {p2cp_rms.mean():.4f} ± {p2cp_rms.std():.4f} px\n")
                    
                    if len(jaccards) > 0:
                        f.write(f"  Jaccard:   {jaccards.mean():.4f} ± {jaccards.std():.4f}\n")
                    
                    f.write(f"  Avg Confidence: {class_data['confidence'].mean():.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            p2cp_means_all = valid_results['p2cp_mean'].dropna()
            p2cp_rms_all = valid_results['p2cp_rms'].dropna()
            jaccards_all = valid_results['jaccard_index'].dropna()
            
            if len(p2cp_means_all) > 0:
                f.write(f"Average P2CP Mean: {p2cp_means_all.mean():.4f} px\n")
                f.write(f"Average P2CP RMS:  {p2cp_rms_all.mean():.4f} px\n")
            
            if len(jaccards_all) > 0:
                f.write(f"Average Jaccard:   {jaccards_all.mean():.4f}\n")
            
            f.write(f"\nAverage Confidence: {valid_results['confidence'].mean():.4f}\n")

        print(f"Summary saved to: {summary_path}")

        # Print summary to console
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        print(f"Total images processed: {df['image_name'].nunique()}")
        print(f"Total predictions: {len(valid_results)}")
        
        if len(p2cp_means_all) > 0:
            print(f"\nAverage P2CP Mean: {p2cp_means_all.mean():.4f} px")
            print(f"Average P2CP RMS:  {p2cp_rms_all.mean():.4f} px")
        
        if len(jaccards_all) > 0:
            print(f"Average Jaccard:   {jaccards_all.mean():.4f}")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Segmentation Inference and Evaluation with Config Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file (recommended)
  python inference_yolo_with_config.py --config config/Nam_exp_01082026/inference_yolo_test.yaml

  # Run single image without config
  python inference_yolo_with_config.py \\
      --weights runs/segment/vocal_tract_yolo26x_1000ep/weights/best.pt \\
      --image data_yolo/images/test/image_001.jpg \\
      --label data_yolo/labels/test/image_001.txt \\
      --output evaluation_results
        """
    )
    
    # Config file argument
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    # Individual arguments (for non-config usage)
    parser.add_argument('--weights', type=str, help='Path to YOLO model weights')
    parser.add_argument('--image', type=str, help='Path to single test image')
    parser.add_argument('--label', type=str, help='Path to ground truth label')
    parser.add_argument('--output', type=str, default='evaluation_results', 
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print("="*80)
        print("YOLO SEGMENTATION EVALUATION - CONFIG MODE")
        print("="*80)
        print(f"\nLoading configuration from: {args.config}\n")
        
        config = load_config(args.config)
        config['_config_path'] = args.config
        
        # Validate config
        required_keys = ['model', 'paths', 'labels', 'evaluation']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
        
        # Load model
        print(f"Loading YOLO model: {config['model']['weights']}")
        model = YOLO(config['model']['weights'])
        
        device = config['model'].get('device', 0)
        if device != 'cpu':
            model.to(f'cuda:{device}')
        
        print(f"✓ Model loaded successfully\n")
        
        # Run batch evaluation
        results = batch_evaluate(model, config)
        
        if results:
            save_results(results, config)
        
    else:
        # Single image mode (backward compatibility)
        print("="*80)
        print("YOLO SEGMENTATION EVALUATION - SINGLE IMAGE MODE")
        print("="*80)
        
        if not args.weights or not args.image or not args.label:
            parser.error("--weights, --image, and --label are required when not using --config")
        
        # Create minimal config
        config = {
            'model': {'weights': args.weights},
            'paths': {'output_folder': args.output},
            'labels': {i: name for i, name in enumerate([
                "arytenoid-cartilage", "epiglottis", "lower-lip", "pharynx",
                "soft-palate-midline", "thyroid-cartilage", "tongue",
                "upper-lip", "vocal-folds"
            ])},
            'evaluation': {
                'closed_articulators': ["tongue", "soft-palate-midline", "epiglottis", "pharynx"],
                'save_visualizations': True,
                'save_per_image': True
            },
            'inference': {'conf_threshold': args.conf},
            'postprocessing': {'regularize_bspline': True, 'bspline_degree': 2, 'bspline_points': 100},
            'visualization': {}
        }
        
        model = YOLO(args.weights)
        print(f"✓ Model loaded\n")
        
        eval_data = evaluate_single_image(model, args.image, args.label, config)
        
        if eval_data:
            os.makedirs(args.output, exist_ok=True)
            
            viz_path = os.path.join(args.output, 
                                   f'eval_{os.path.splitext(os.path.basename(args.image))[0]}.png')
            visualize_results(
                eval_data['img'],
                eval_data['pred_masks'],
                eval_data['gt_masks'],
                eval_data['results'],
                viz_path,
                config
            )
            
            print(f"\n✓ Results saved to: {args.output}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
