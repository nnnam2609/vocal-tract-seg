"""
YOLO Segmentation Inference and Evaluation Script
Performs inference on a test image, calculates metrics (P2CP, Jaccard), and visualizes results
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import splprep, splev
from ultralytics import YOLO
import yaml


# Define class mapping based on data.yaml
CLASS_NAMES = {
    0: "arytenoid-cartilage",
    1: "epiglottis",
    2: "lower-lip",
    3: "pharynx",
    4: "soft-palate-midline",
    5: "thyroid-cartilage",
    6: "tongue",
    7: "upper-lip",
    8: "vocal-folds"
}

# Closed articulators (for Jaccard calculation) - adjust based on your domain knowledge
CLOSED_ARTICULATORS = ["tongue", "soft-palate-midline", "epiglottis", "pharynx"]


def extract_contour_from_mask(binary_mask):
    """Extract contour from binary mask using OpenCV"""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
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
    """Regularize contour using B-spline interpolation"""
    if contour is None or len(contour) < 4:
        return None
    
    try:
        x, y = contour[:, 0], contour[:, 1]
        tck, u = splprep([x, y], s=0, k=min(degree, len(contour)-1))
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.array([x_new, y_new]).T
    except:
        return contour


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
        return None, None
    
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


def evaluate_single_image(model, image_path, label_path, output_dir, conf_threshold=0.25):
    """
    Perform inference and evaluation on a single image
    
    Args:
        model: YOLO model
        image_path: Path to test image
        label_path: Path to ground truth label (YOLO format)
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        dict: Evaluation metrics for each class
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING IMAGE: {os.path.basename(image_path)}")
    print(f"{'='*80}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Run inference
    print("Running YOLO inference...")
    results = model.predict(
        image_path, 
        conf=conf_threshold,
        verbose=False
    )[0]
    
    # Load ground truth
    print("Loading ground truth labels...")
    gt_masks = load_ground_truth_mask(label_path, img.shape)
    
    if gt_masks is None:
        print(f" No ground truth found at: {label_path}")
        return None
    
    # Extract predictions
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
    
    print(f"Found {len(pred_masks)} predicted classes")
    print(f"Found {len(gt_masks)} ground truth classes")
    
    # Calculate metrics for each class
    evaluation_results = []
    
    for class_id in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_id]
        
        result = {
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
            # Extract contours
            pred_contour = extract_contour_from_mask(pred_mask)
            gt_contour = extract_contour_from_mask(gt_mask)
            
            if pred_contour is None or gt_contour is None:
                print(f"{class_name}: Could not extract contours")
                evaluation_results.append(result)
                continue
            
            # Regularize contours
            reg_pred = regularize_contour_bspline(pred_contour, degree=2, num_points=100)
            reg_gt = regularize_contour_bspline(gt_contour, degree=2, num_points=100)
            
            if reg_pred is None or reg_gt is None:
                reg_pred = pred_contour
                reg_gt = gt_contour
            
            # Calculate P2CP metrics
            result['p2cp_mean'] = p2cp_mean_distance(reg_pred, reg_gt)
            result['p2cp_rms'] = p2cp_rms_distance(reg_pred, reg_gt)
            
            # Calculate Jaccard for closed articulators
            if class_name in CLOSED_ARTICULATORS:
                pred_filled = create_filled_mask_from_contour(pred_contour, img.shape[:2])
                gt_filled = create_filled_mask_from_contour(gt_contour, img.shape[:2])
                result['jaccard_index'] = jaccard_index(pred_filled, gt_filled)
            
            print(f"{class_name}: P2CP_mean={result['p2cp_mean']:.4f}px, "
                  f"P2CP_rms={result['p2cp_rms']:.4f}px" + 
                  (f", Jaccard={result['jaccard_index']:.4f}" if not np.isnan(result['jaccard_index']) else ""))
            
        except Exception as e:
            print(f"{class_name}: Error - {str(e)}")
        
        evaluation_results.append(result)
    
    # Visualize results
    print(f"\nCreating visualization...")
    visualize_results(img_rgb, pred_masks, gt_masks, evaluation_results, output_dir, 
                      os.path.basename(image_path))
    
    return evaluation_results


def visualize_results(img, pred_masks, gt_masks, results, output_dir, image_name):
    """Create comprehensive visualization of results"""
    
    h, w = img.shape[:2]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'YOLO Segmentation Evaluation - {image_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Ground Truth Overlay
    gt_overlay = img.copy()
    gt_composite = np.zeros((h, w), dtype=np.uint8)
    for class_id, mask in gt_masks.items():
        gt_composite[mask > 0] = class_id + 1
    
    axes[0, 1].imshow(img, alpha=0.6)
    axes[0, 1].imshow(gt_composite, cmap='tab10', alpha=0.5, vmin=0, vmax=10)
    axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Prediction Overlay
    pred_composite = np.zeros((h, w), dtype=np.uint8)
    for class_id, pred_data in pred_masks.items():
        mask = pred_data['mask']
        pred_composite[mask > 0] = class_id + 1
    
    axes[0, 2].imshow(img, alpha=0.6)
    axes[0, 2].imshow(pred_composite, cmap='tab10', alpha=0.5, vmin=0, vmax=10)
    axes[0, 2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. Ground Truth Contours
    axes[1, 0].imshow(img, alpha=0.5)
    for class_id, mask in gt_masks.items():
        contour = extract_contour_from_mask(mask)
        if contour is not None:
            axes[1, 0].plot(contour[:, 0], contour[:, 1], linewidth=2.5, 
                           label=CLASS_NAMES[class_id])
    axes[1, 0].set_title('Ground Truth Contours', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='upper right', fontsize=7)
    axes[1, 0].axis('off')
    
    # 5. Predicted Contours
    axes[1, 1].imshow(img, alpha=0.5)
    for class_id, pred_data in pred_masks.items():
        mask = pred_data['mask']
        contour = extract_contour_from_mask(mask)
        if contour is not None:
            conf = pred_data['conf']
            axes[1, 1].plot(contour[:, 0], contour[:, 1], linewidth=2.5,
                           label=f"{CLASS_NAMES[class_id]} ({conf:.2f})")
    axes[1, 1].set_title('Predicted Contours', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='upper right', fontsize=7)
    axes[1, 1].axis('off')
    
    # 6. Metrics Table
    axes[1, 2].axis('off')
    
    # Create metrics text
    metrics_text = "EVALUATION METRICS\n" + "="*40 + "\n\n"
    
    valid_results = [r for r in results if r['has_prediction'] and r['has_ground_truth']]
    
    if valid_results:
        for r in valid_results:
            metrics_text += f"{r['class_name'][:15]:15s}\n"
            metrics_text += f"  P2CP Mean:  {r['p2cp_mean']:.4f} px\n"
            metrics_text += f"  P2CP RMS:   {r['p2cp_rms']:.4f} px\n"
            if not np.isnan(r['jaccard_index']):
                metrics_text += f"  Jaccard:    {r['jaccard_index']:.4f}\n"
            metrics_text += f"  Conf:       {r['confidence']:.4f}\n\n"
        
        # Overall statistics
        metrics_text += "\n" + "="*40 + "\n"
        metrics_text += "OVERALL STATISTICS\n" + "="*40 + "\n"
        
        p2cp_means = [r['p2cp_mean'] for r in valid_results if not np.isnan(r['p2cp_mean'])]
        p2cp_rms = [r['p2cp_rms'] for r in valid_results if not np.isnan(r['p2cp_rms'])]
        jaccards = [r['jaccard_index'] for r in valid_results if not np.isnan(r['jaccard_index'])]
        
        if p2cp_means:
            metrics_text += f"\nAvg P2CP Mean: {np.mean(p2cp_means):.4f} px\n"
            metrics_text += f"Avg P2CP RMS:  {np.mean(p2cp_rms):.4f} px\n"
        if jaccards:
            metrics_text += f"Avg Jaccard:   {np.mean(jaccards):.4f}\n"
        
        metrics_text += f"\nClasses eval:  {len(valid_results)}/{len(CLASS_NAMES)}\n"
    else:
        metrics_text += "\n No valid predictions to evaluate"
    
    axes[1, 2].text(0.05, 0.95, metrics_text, 
                    transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                             edgecolor='black', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'evaluation_{image_name.rsplit(".", 1)[0]}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_path}")
    plt.show()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Segmentation Inference and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single test image
  python inference_yolo_eval.py \\
      --weights runs/segment/vocal_tract_yolo26x_1000ep/weights/best.pt \\
      --image data_yolo/images/test/image_001.jpg \\
      --label data_yolo/labels/test/image_001.txt \\
      --output evaluation_results

  # With custom confidence threshold
  python inference_yolo_eval.py \\
      --weights runs/segment/vocal_tract_yolo26x_1000ep/weights/best.pt \\
      --image data_yolo/images/test/image_001.jpg \\
      --label data_yolo/labels/test/image_001.txt \\
      --conf 0.5 \\
      --output evaluation_results
        """
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to YOLO model weights (e.g., best.pt)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to test image'
    )
    
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help='Path to ground truth label file (YOLO format .txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for predictions (default: 0.25)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    if not os.path.exists(args.label):
        raise FileNotFoundError(f"Label file not found: {args.label}")
    
    print("="*80)
    print("YOLO SEGMENTATION INFERENCE AND EVALUATION")
    print("="*80)
    print(f"\nModel weights: {args.weights}")
    print(f"Test image:    {args.image}")
    print(f"Ground truth:  {args.label}")
    print(f"Output dir:    {args.output}")
    print(f"Conf thresh:   {args.conf}")
    
    # Load model
    print(f"\nLoading YOLO model...")
    model = YOLO(args.weights)
    print(f"✓ Model loaded successfully")
    
    # Run evaluation
    results = evaluate_single_image(
        model=model,
        image_path=args.image,
        label_path=args.label,
        output_dir=args.output,
        conf_threshold=args.conf
    )
    
    if results:
        # Save metrics to file
        metrics_file = os.path.join(args.output, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"YOLO Segmentation Evaluation Results\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Image: {os.path.basename(args.image)}\n")
            f.write(f"Model: {args.weights}\n\n")
            f.write(f"{'Class':<20} {'P2CP Mean':>12} {'P2CP RMS':>12} {'Jaccard':>12} {'Conf':>8}\n")
            f.write(f"{'-'*80}\n")
            
            for r in results:
                if r['has_prediction'] and r['has_ground_truth']:
                    f.write(f"{r['class_name']:<20} ")
                    f.write(f"{r['p2cp_mean']:>12.4f} ")
                    f.write(f"{r['p2cp_rms']:>12.4f} ")
                    f.write(f"{r['jaccard_index']:>12.4f} " if not np.isnan(r['jaccard_index']) else f"{'N/A':>12} ")
                    f.write(f"{r['confidence']:>8.4f}\n")
        
        print(f"\n✓ Metrics saved to: {metrics_file}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
