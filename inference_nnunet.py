#!/usr/bin/env python3
"""
nnUNet Inference Script with Config File Support
Usage: python inference_nnunet.py --config config/Nam_exp_01082026/inference_nnunet.yaml
"""

import argparse
import yaml
import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import splprep, splev

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


def load_config(config_path):
    """Load and validate YAML config"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Replace environment variables and expand paths
    if 'paths' in cfg:
        for key, value in cfg['paths'].items():
            if isinstance(value, str):
                cfg['paths'][key] = os.path.expandvars(os.path.expanduser(value))
    
    return cfg


def initialize_predictor(cfg):
    """Initialize nnUNet predictor from config"""
    model_folder = os.path.join(
        cfg['paths']['nnunet_results'],
        cfg['model']['dataset_name'],
        f"{cfg['model']['trainer']}__{cfg['model']['plans']}__{cfg['model']['configuration']}"
    )
    
    print(f"Model folder: {model_folder}")
    
    if not os.path.exists(model_folder):
        raise ValueError(f"Model folder not found: {model_folder}")
    
    # Determine device
    device_config = cfg['inference']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")
    
    predictor = nnUNetPredictor(
        tile_step_size=cfg['inference']['tile_step_size'],
        use_gaussian=cfg['inference']['use_gaussian'],
        use_mirroring=cfg['inference']['use_mirroring'],
        device=device,
        verbose=cfg['inference'].get('verbose', True),
        verbose_preprocessing=cfg['inference'].get('verbose_preprocessing', False),
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(cfg['model']['fold'],),
        checkpoint_name=cfg['model']['checkpoint']
    )
    
    return predictor


def collect_input_images(cfg):
    """Collect images to process based on config"""
    input_folder = cfg['paths']['input_folder']
    
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")
    
    if 'images' in cfg['data'] and cfg['data']['images']:
        # Specific images
        images = cfg['data']['images']
    elif 'image_range' in cfg['data'] and cfg['data']['image_range']:
        # Range of images
        start, end = cfg['data']['image_range']
        images = [f"case_{i:05d}_0000.nii.gz" for i in range(start, end)]
    elif cfg['data'].get('process_all', False):
        # All images
        images = sorted([f for f in os.listdir(input_folder) if f.endswith('.nii.gz')])
    else:
        raise ValueError("Must specify images, image_range, or process_all in config")
    
    # Filter existing images
    image_paths = []
    for img in images:
        img_path = os.path.join(input_folder, img)
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"⚠️  Image not found: {img}")
    
    return image_paths


def run_inference(predictor, image_paths, output_folder, cfg):
    """Run inference on images"""
    os.makedirs(output_folder, exist_ok=True)
    
    reader = SimpleITKIO()
    writer = SimpleITKIO()
    
    results = []
    
    for img_path in tqdm(image_paths, desc="Running inference"):
        img_name = os.path.basename(img_path)
        
        try:
            # Load input
            input_array, properties = reader.read_images([img_path])
            
            # Predict
            prediction = predictor.predict_from_list_of_npy_arrays(
                [input_array],
                [None],
                [properties],
                None,
                num_processes=cfg['inference']['num_processes'],
                save_probabilities=cfg['inference']['save_probabilities'],
                num_processes_segmentation_export=1
            )[0]
            
            # Save prediction
            output_path = os.path.join(output_folder, img_name)
            writer.write_seg(prediction, output_path, properties)
            
            results.append({
                'image': img_name,
                'input_path': img_path,
                'output_path': output_path,
                'labels': np.unique(prediction).tolist(),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  ✗ Error processing {img_name}: {e}")
            results.append({
                'image': img_name,
                'input_path': img_path,
                'output_path': None,
                'labels': [],
                'status': 'failed',
                'error': str(e)
            })
    
    return results


# Evaluation functions
def extract_contour_from_mask(binary_mask):
    """Extract contour from binary mask using OpenCV"""
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_NONE)
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


def create_filled_mask_from_contour(contour, shape=(136, 136)):
    """Create a filled binary mask from contour"""
    mask = np.zeros(shape, dtype=np.uint8)
    if contour is None or len(contour) < 3:
        return mask
    
    contour_int = contour.astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 1)
    mask = binary_fill_holes(mask).astype(int)
    return mask


def create_comparison_visualization(input_img, contours_dict, img_name, vis_folder, label_to_class):
    """
    Create a 3-panel comparison visualization:
    1. Ground truth contours on original image
    2. Prediction contours on original image
    3. Ground truth vs Prediction contours overlaid
    """
    # Define colormap for different articulators
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    class_colors = {class_name: colors[i] for i, class_name in enumerate(label_to_class.values())}
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Ground Truth on Original
    axes[0].imshow(input_img, cmap='gray')
    for class_name, contour in contours_dict['gt'].items():
        if contour is not None and len(contour) > 0:
            axes[0].plot(contour[:, 0], contour[:, 1], 
                        color=class_colors[class_name], 
                        linewidth=2, 
                        label=class_name,
                        alpha=0.9)
    axes[0].set_title('Ground Truth Contours', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    # Panel 2: Prediction on Original
    axes[1].imshow(input_img, cmap='gray')
    for class_name, contour in contours_dict['pred'].items():
        if contour is not None and len(contour) > 0:
            axes[1].plot(contour[:, 0], contour[:, 1],
                        color=class_colors[class_name],
                        linewidth=2,
                        label=class_name,
                        alpha=0.9)
    axes[1].set_title('Prediction Contours', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    # Panel 3: GT (solid) vs Pred (dashed) comparison
    axes[2].imshow(input_img, cmap='gray')
    for class_name in contours_dict['gt'].keys():
        gt_contour = contours_dict['gt'].get(class_name)
        pred_contour = contours_dict['pred'].get(class_name)
        
        if gt_contour is not None and len(gt_contour) > 0:
            axes[2].plot(gt_contour[:, 0], gt_contour[:, 1],
                        color=class_colors[class_name],
                        linewidth=2.5,
                        linestyle='-',
                        alpha=0.8,
                        label=f'{class_name} (GT)')
        
        if pred_contour is not None and len(pred_contour) > 0:
            axes[2].plot(pred_contour[:, 0], pred_contour[:, 1],
                        color=class_colors[class_name],
                        linewidth=2,
                        linestyle='--',
                        alpha=0.7,
                        label=f'{class_name} (Pred)')
    
    axes[2].set_title('GT (solid) vs Prediction (dashed)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    axes[2].legend(loc='upper right', fontsize=7, framealpha=0.8, ncol=2)
    
    plt.suptitle(f'Contour Comparison: {img_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    case_name = img_name.replace('_0000.nii.gz', '').replace('.nii.gz', '')
    save_path = os.path.join(vis_folder, f'{case_name}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def evaluate_predictions(cfg, inference_results):
    """Evaluate predictions if ground truth available"""
    if not cfg['evaluation']['enabled']:
        print("Evaluation disabled in config")
        return None
    
    gt_folder = cfg['paths'].get('ground_truth_folder')
    if gt_folder is None or not os.path.exists(gt_folder):
        print(f"⚠️  Ground truth folder not found: {gt_folder}")
        return None
    
    print(f"\nGround truth folder: {gt_folder}")
    
    label_to_class = cfg['labels']
    closed_articulators = cfg['evaluation']['closed_articulators']
    save_visualizations = cfg['evaluation'].get('save_visualizations', False)
    save_contours = cfg['postprocessing'].get('save_contours', False)
    extract_contours_flag = cfg['postprocessing'].get('extract_contours', False)
    regularize_flag = cfg['postprocessing'].get('regularize_bspline', False)
    
    # Create directories
    output_folder = cfg['paths']['output_folder']
    if save_visualizations:
        vis_folder = os.path.join(output_folder, 'visualizations')
        os.makedirs(vis_folder, exist_ok=True)
    if save_contours:
        contours_folder = os.path.join(output_folder, 'contours')
        os.makedirs(contours_folder, exist_ok=True)
    
    eval_results = []
    
    successful_results = [r for r in inference_results if r['status'] == 'success']
    
    for result in tqdm(successful_results, desc="Evaluating"):
        img_name = result['image']
        gt_path = os.path.join(gt_folder, img_name.replace('_0000.nii.gz', '.nii.gz'))
        
        if not os.path.exists(gt_path):
            print(f"  ⚠️  Ground truth not found for {img_name}")
            continue
        
        try:
            # Load masks
            pred_nii = nib.load(result['output_path'])
            gt_nii = nib.load(gt_path)
            input_nii = nib.load(result['input_path'])
            
            pred_mask = pred_nii.get_fdata().squeeze()
            gt_mask = gt_nii.get_fdata().squeeze()
            input_img = input_nii.get_fdata().squeeze()
            
            # Store contours for visualization
            image_contours = {'pred': {}, 'gt': {}}
            
            # Evaluate per class
            for label, class_name in label_to_class.items():
                eval_item = {
                    'image': img_name,
                    'class': class_name,
                    'label': label,
                    'p2cp_rms': np.nan,
                    'p2cp_mean': np.nan,
                    'jaccard_index': np.nan,
                    'has_prediction': False,
                    'has_ground_truth': False,
                    'pred_pixels': 0,
                    'gt_pixels': 0
                }
                
                # Extract binary masks for this class
                pred_binary = (pred_mask == label).astype(np.uint8) * 255
                gt_binary = (gt_mask == label).astype(np.uint8) * 255
                
                pred_pixel_count = (pred_binary > 0).sum()
                gt_pixel_count = (gt_binary > 0).sum()
                
                eval_item['pred_pixels'] = int(pred_pixel_count)
                eval_item['gt_pixels'] = int(gt_pixel_count)
                
                if gt_pixel_count == 0:
                    eval_results.append(eval_item)
                    continue
                
                eval_item['has_ground_truth'] = True
                
                if pred_pixel_count == 0:
                    eval_results.append(eval_item)
                    continue
                
                eval_item['has_prediction'] = True
                
                try:
                    # Extract contours
                    pred_contour = extract_contour_from_mask(pred_binary)
                    gt_contour = extract_contour_from_mask(gt_binary)
                    
                    if pred_contour is None or gt_contour is None:
                        eval_results.append(eval_item)
                        continue
                    
                    # Regularize contours with B-splines if enabled
                    if regularize_flag:
                        reg_pred = regularize_contour_bspline(
                            pred_contour, 
                            degree=cfg['postprocessing']['bspline_degree'],
                            num_points=cfg['postprocessing']['bspline_points']
                        )
                        reg_gt = regularize_contour_bspline(
                            gt_contour,
                            degree=cfg['postprocessing']['bspline_degree'],
                            num_points=cfg['postprocessing']['bspline_points']
                        )
                        
                        if reg_pred is None:
                            reg_pred = pred_contour
                        if reg_gt is None:
                            reg_gt = gt_contour
                    else:
                        reg_pred = pred_contour
                        reg_gt = gt_contour
                    
                    # Store contours for visualization and saving
                    image_contours['pred'][class_name] = reg_pred
                    image_contours['gt'][class_name] = reg_gt
                    
                    # Save contours as .npy files if enabled
                    if save_contours:
                        case_name = img_name.replace('_0000.nii.gz', '').replace('.nii.gz', '')
                        pred_contour_path = os.path.join(
                            contours_folder, 
                            f"{case_name}_{class_name}_pred.npy"
                        )
                        gt_contour_path = os.path.join(
                            contours_folder,
                            f"{case_name}_{class_name}_gt.npy"
                        )
                        np.save(pred_contour_path, reg_pred)
                        np.save(gt_contour_path, reg_gt)
                    
                    # Calculate P2CP metrics
                    eval_item['p2cp_mean'] = p2cp_mean_distance(reg_pred, reg_gt)
                    eval_item['p2cp_rms'] = p2cp_rms_distance(reg_pred, reg_gt)
                    
                    # Calculate Jaccard for closed articulators
                    if class_name in closed_articulators:
                        pred_filled = create_filled_mask_from_contour(pred_contour, shape=pred_mask.shape)
                        gt_filled = create_filled_mask_from_contour(gt_contour, shape=gt_mask.shape)
                        eval_item['jaccard_index'] = jaccard_index(pred_filled, gt_filled)
                    
                except Exception as e:
                    print(f"    ⚠️  Error evaluating {class_name} in {img_name}: {e}")
                
                eval_results.append(eval_item)
            
            # Create visualization if enabled
            if save_visualizations and len(image_contours['pred']) > 0:
                try:
                    create_comparison_visualization(
                        input_img, 
                        image_contours,
                        img_name,
                        vis_folder,
                        label_to_class
                    )
                except Exception as e:
                    print(f"  ⚠️  Error creating visualization for {img_name}: {e}")
                
        except Exception as e:
            print(f"  ✗ Error evaluating {img_name}: {e}")
    
    if not eval_results:
        print("No evaluation results generated")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(eval_results)
    
    # Try to load case mapping if available
    mapping_file = cfg.get('case_mapping_file')
    has_mapping = False
    
    if mapping_file and os.path.exists(mapping_file):
        print(f"\nLoading case mapping from: {mapping_file}")
        df_mapping = pd.read_csv(mapping_file)
        
        # Extract case_name from image column
        # Image format: "case_00897_0000.nii.gz"
        # Mapping format: "case_00897_0000"
        # Remove only the .nii.gz extension
        df['case_name'] = df['image'].str.replace('.nii.gz', '')
        
        # Merge with mapping
        df = df.merge(df_mapping[['case_name', 'subject', 'sequence', 'frame']], 
                     on='case_name', 
                     how='left')
        
        # Reorder columns to have subject/sequence/frame after image
        cols = ['image', 'subject', 'sequence', 'frame'] + [c for c in df.columns if c not in ['image', 'subject', 'sequence', 'frame', 'case_name']]
        df = df[cols]
        
        has_mapping = True
        print(f"✓ Added subject/sequence/frame metadata from mapping file")
    
    # Save results
    if cfg['evaluation']['save_csv']:
        csv_path = os.path.join(cfg['paths']['output_folder'], 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Evaluation saved to: {csv_path}")
    
    # Generate Mask R-CNN compatible format if mapping is available
    if has_mapping and cfg['evaluation'].get('save_maskrcnn_format', True):
        print(f"\nGenerating Mask R-CNN compatible output...")
        
        # Filter only rows with predictions and ground truth
        df_valid = df[df['has_prediction'] & df['has_ground_truth']].copy()
        
        if len(df_valid) > 0:
            # Create output in Mask R-CNN format
            # Columns: subject, sequence, frame, pred_class, p2cp_rms, p2cp_mean, jaccard_index
            df_maskrcnn = df_valid[['subject', 'sequence', 'frame', 'class', 'p2cp_rms', 'p2cp_mean', 'jaccard_index']].copy()
            df_maskrcnn.rename(columns={'class': 'pred_class'}, inplace=True)
            
            # Sort by subject, sequence, frame, pred_class for consistency
            df_maskrcnn.sort_values(['subject', 'sequence', 'frame', 'pred_class'], inplace=True)
            
            # Save to CSV
            maskrcnn_csv_path = os.path.join(cfg['paths']['output_folder'], 'test_results_nnunet.csv')
            df_maskrcnn.to_csv(maskrcnn_csv_path, index=False)
            print(f"✓ Mask R-CNN format saved to: {maskrcnn_csv_path}")
            print(f"  Format: subject, sequence, frame, pred_class, p2cp_rms, p2cp_mean, jaccard_index")
            print(f"  Total entries: {len(df_maskrcnn)}")
        else:
            print("⚠️  No valid predictions with ground truth found for Mask R-CNN format")
    
    # Print summary
    df_valid = df[df['has_prediction'] & df['has_ground_truth']]
    
    if len(df_valid) > 0:
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Classes evaluated: {len(df_valid)}/{len(df)}")
        print(f"\nPoint-to-Curve Metrics (pixels):")
        print(f"  Mean P2CP RMS:   {df_valid['p2cp_rms'].mean():.4f}")
        print(f"  Mean P2CP Mean:  {df_valid['p2cp_mean'].mean():.4f}")
        
        jaccard_valid = df_valid[~df_valid['jaccard_index'].isna()]
        if len(jaccard_valid) > 0:
            print(f"\nJaccard Index (IoU) for closed articulators:")
            print(f"  Mean Jaccard:    {jaccard_valid['jaccard_index'].mean():.4f}")
        print("="*70)
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description='nnUNet Inference with Config File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    
    # Set environment variables
    if 'nnunet_results' in cfg['paths']:
        os.environ['nnUNet_results'] = cfg['paths']['nnunet_results']
    if 'nnunet_raw' in cfg['paths']:
        os.environ['nnUNet_raw'] = cfg['paths']['nnunet_raw']
    if 'nnunet_preprocessed' in cfg['paths']:
        os.environ['nnUNet_preprocessed'] = cfg['paths']['nnunet_preprocessed']
    
    print("\n" + "="*70)
    print("nnUNet CONFIG-BASED INFERENCE")
    print("="*70)
    print(f"Dataset: {cfg['model']['dataset_name']}")
    print(f"Configuration: {cfg['model']['configuration']}")
    print(f"Fold: {cfg['model']['fold']}")
    print(f"Output: {cfg['paths']['output_folder']}")
    print("="*70 + "\n")
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = initialize_predictor(cfg)
    print("✓ Model loaded\n")
    
    # Collect images
    print("Collecting images...")
    image_paths = collect_input_images(cfg)
    print(f"✓ Found {len(image_paths)} images to process\n")
    
    if len(image_paths) == 0:
        print("No images to process. Exiting.")
        return
    
    # Run inference
    print("Running inference...")
    results = run_inference(predictor, image_paths, cfg['paths']['output_folder'], cfg)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✓ Inference complete: {successful}/{len(results)} successful\n")
    
    # Evaluate
    if cfg['evaluation']['enabled']:
        print("Evaluating predictions...")
        eval_results = evaluate_predictions(cfg, results)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
