"""
YOLO Evaluation Script
Evaluates YOLO segmentation model on test set and calculates metrics:
- P2CP (Point-to-Closest-Point) distance: mean and RMS
- Jaccard Index (IoU) for closed articulators
- Dice coefficient (optional)

Configuration driven - see config/yolo_evaluation.yaml
"""

import argparse
import os
import numpy as np
import cv2
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import splprep, splev


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data_yaml(data_yaml_path):
    """Load YOLO data.yaml to get class names"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config


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


def dice_coefficient(pred_mask, gt_mask, eps=1e-15):
    """Calculate Dice coefficient"""
    intersection = (pred_mask * gt_mask).sum()
    return (2 * intersection + eps) / (pred_mask.sum() + gt_mask.sum() + eps)


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
    """Load YOLO format label and convert to segmentation masks"""
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
            coords = []
            
            # Parse normalized coordinates
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    x = float(parts[i]) * w
                    y = float(parts[i+1]) * h
                    coords.append([int(x), int(y)])
            
            if len(coords) > 0:
                coords = np.array(coords, dtype=np.int32)
                
                # Create mask for this class
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [coords], 255)
                masks[class_id] = mask
    
    return masks


def evaluate_single_image(model, image_path, label_path, config, class_id_to_name):
    """
    Perform inference and evaluation on a single image
    
    Returns:
        list: Evaluation results for each class
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Run inference
    conf_threshold = config['model']['conf_threshold']
    results = model.predict(
        str(image_path), 
        conf=conf_threshold,
        verbose=False
    )[0]
    
    # Load ground truth
    gt_masks = load_ground_truth_mask(label_path, img.shape)
    
    if gt_masks is None:
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
    
    # Calculate metrics for each class
    evaluation_results = []
    closed_articulators = config.get('closed_articulators', [])
    reg_config = config.get('regularization', {})
    
    for class_id in range(len(class_id_to_name)):
        class_name = class_id_to_name.get(class_id, f"class_{class_id}")
        
        result = {
            'image': os.path.basename(image_path),
            'class_id': class_id,
            'class_name': class_name,
            'p2cp_mean': np.nan,
            'p2cp_rms': np.nan,
            'jaccard_index': np.nan,
            'dice_coeff': np.nan,
            'has_prediction': class_id in pred_masks,
            'has_ground_truth': class_id in gt_masks,
            'confidence': pred_masks[class_id]['conf'] if class_id in pred_masks else 0.0,
        }
        
        # Skip if no ground truth for this class
        if class_id not in gt_masks:
            evaluation_results.append(result)
            continue
        
        gt_mask = gt_masks[class_id]
        
        # Skip if no prediction for this class
        if class_id not in pred_masks:
            evaluation_results.append(result)
            continue
        
        pred_mask = pred_masks[class_id]['mask']
        
        try:
            # Extract contours
            pred_contour = extract_contour_from_mask(pred_mask)
            gt_contour = extract_contour_from_mask(gt_mask)
            
            if pred_contour is None or gt_contour is None:
                evaluation_results.append(result)
                continue
            
            # Regularize contours if enabled
            if reg_config.get('enabled', True) and reg_config.get('method') == 'bspline':
                reg_pred = regularize_contour_bspline(
                    pred_contour,
                    degree=reg_config.get('degree', 2),
                    num_points=reg_config.get('num_points', 100)
                )
                reg_gt = regularize_contour_bspline(
                    gt_contour,
                    degree=reg_config.get('degree', 2),
                    num_points=reg_config.get('num_points', 100)
                )
                
                if reg_pred is None or reg_gt is None:
                    reg_pred = pred_contour
                    reg_gt = gt_contour
            else:
                reg_pred = pred_contour
                reg_gt = gt_contour
            
            # Calculate P2CP metrics
            if config['metrics'].get('p2cp', True):
                result['p2cp_mean'] = p2cp_mean_distance(reg_pred, reg_gt)
                result['p2cp_rms'] = p2cp_rms_distance(reg_pred, reg_gt)
            
            # Calculate Jaccard for closed articulators
            if config['metrics'].get('jaccard', True) and class_name in closed_articulators:
                pred_filled = create_filled_mask_from_contour(pred_contour, img.shape[:2])
                gt_filled = create_filled_mask_from_contour(gt_contour, img.shape[:2])
                result['jaccard_index'] = jaccard_index(pred_filled, gt_filled)
            
            # Calculate Dice coefficient
            if config['metrics'].get('dice', False):
                pred_filled = create_filled_mask_from_contour(pred_contour, img.shape[:2])
                gt_filled = create_filled_mask_from_contour(gt_contour, img.shape[:2])
                result['dice_coeff'] = dice_coefficient(pred_filled, gt_filled)
            
        except Exception as e:
            print(f"\nError processing {class_name}: {str(e)}")
        
        evaluation_results.append(result)
    
    return evaluation_results


def run_evaluation(config):
    """Run evaluation on test set"""
    
    # Load model
    model_path = config['model']['weights']
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Load data configuration
    data_yaml_path = config['data']['data_yaml']
    data_config = load_data_yaml(data_yaml_path)
    class_id_to_name = data_config.get('names', {})
    
    # Get test directories
    data_root = Path(data_yaml_path).parent
    split = config['data']['split']
    test_img_dir = data_root / f"images/{split}"
    test_label_dir = data_root / f"labels/{split}"
    
    if not test_img_dir.exists():
        raise ValueError(f"Test image directory not found: {test_img_dir}")
    if not test_label_dir.exists():
        raise ValueError(f"Test label directory not found: {test_label_dir}")
    
    # Get all test images
    test_images = sorted(list(test_img_dir.glob('*.png')))
    print(f"Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"\nEvaluating images...")
    
    # Evaluate each image
    all_results = []
    
    for image_path in tqdm(test_images, desc="Evaluating"):
        # Get corresponding label file
        label_path = test_label_dir / (image_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # Evaluate single image
        results = evaluate_single_image(model, image_path, label_path, config, class_id_to_name)
        
        if results:
            all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    print("Metrics by class:")
    print("-" * 80)
    
    for class_name in sorted(df['class_name'].unique()):
        class_df = df[df['class_name'] == class_name]
        
        # Filter only rows with predictions and ground truth
        valid_df = class_df[(class_df['has_prediction']) & (class_df['has_ground_truth'])]
        
        if len(valid_df) == 0:
            continue
        
        print(f"\n{class_name}:")
        print(f"  Samples: {len(valid_df)}")
        
        if config['metrics'].get('p2cp', True):
            p2cp_mean_avg = valid_df['p2cp_mean'].mean()
            p2cp_rms_avg = valid_df['p2cp_rms'].mean()
            print(f"  P2CP Mean: {p2cp_mean_avg:.4f} ± {valid_df['p2cp_mean'].std():.4f} px")
            print(f"  P2CP RMS:  {p2cp_rms_avg:.4f} ± {valid_df['p2cp_rms'].std():.4f} px")
        
        if config['metrics'].get('jaccard', True):
            jaccard_valid = valid_df[~valid_df['jaccard_index'].isna()]
            if len(jaccard_valid) > 0:
                print(f"  Jaccard:   {jaccard_valid['jaccard_index'].mean():.4f} ± {jaccard_valid['jaccard_index'].std():.4f}")
        
        if config['metrics'].get('dice', False):
            dice_valid = valid_df[~valid_df['dice_coeff'].isna()]
            if len(dice_valid) > 0:
                print(f"  Dice:      {dice_valid['dice_coeff'].mean():.4f} ± {dice_valid['dice_coeff'].std():.4f}")
    
    # Save detailed results as CSV
    if config['output'].get('save_csv', True):
        csv_file = output_dir / 'evaluation_results_detailed.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n\nDetailed results saved to: {csv_file}")
    
    # Save summary statistics
    if config['output'].get('save_summary', True):
        summary_file = output_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("YOLO Evaluation Summary\n")
            f.write("="*80 + "\n\n")
            
            for class_name in sorted(df['class_name'].unique()):
                class_df = df[df['class_name'] == class_name]
                valid_df = class_df[(class_df['has_prediction']) & (class_df['has_ground_truth'])]
                
                if len(valid_df) == 0:
                    continue
                
                f.write(f"{class_name}:\n")
                f.write(f"  Samples: {len(valid_df)}\n")
                
                if config['metrics'].get('p2cp', True):
                    f.write(f"  P2CP Mean: {valid_df['p2cp_mean'].mean():.4f} ± {valid_df['p2cp_mean'].std():.4f} px\n")
                    f.write(f"  P2CP RMS:  {valid_df['p2cp_rms'].mean():.4f} ± {valid_df['p2cp_rms'].std():.4f} px\n")
                
                if config['metrics'].get('jaccard', True):
                    jaccard_valid = valid_df[~valid_df['jaccard_index'].isna()]
                    if len(jaccard_valid) > 0:
                        f.write(f"  Jaccard:   {jaccard_valid['jaccard_index'].mean():.4f} ± {jaccard_valid['jaccard_index'].std():.4f}\n")
                
                f.write("\n")
        
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='YOLO evaluation script')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to evaluation configuration YAML file')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise ValueError(f"Configuration file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Run evaluation
    run_evaluation(config)


if __name__ == '__main__':
    main()
