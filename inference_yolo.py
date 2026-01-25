"""
YOLO Unified Inference Script
Performs inference on test images with flexible output options:
- Save contours as .npy files (for comparison visualization)
- Save masks as .npy files
- Save visualization images

Configuration driven - see config/yolo_inference.yaml
"""

import argparse
import os
import numpy as np
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker.postprocessing import POST_PROCESSING
from vt_tracker.postprocessing.calculate_contours import calculate_contour


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


def smooth_contour(contour):
    res_x, res_y = regularize_Bsplines(contour, 3)
    return np.column_stack([res_x, res_y])


def extract_contour_from_mask(binary_mask, class_name):
    """Extract Mask-RCNN-like centerline contour from a binary mask."""
    if class_name not in POST_PROCESSING:
        return None

    mask = binary_mask.astype(np.float32) / 255.0
    cfg = dict(POST_PROCESSING[class_name])
    contour = calculate_contour(class_name, mask, cfg=cfg)
    if contour is None or len(contour) == 0:
        return None

    contour = smooth_contour(contour)
    if len(contour.shape) == 1:  # Single point
        return None

    return contour


def parse_image_path(image_path):
    """
    Parse image path to extract dataset, subject, sequence, and frame
    Example: data_yolo/test/images/ArtSpeech_Vocal_Tract_Segmentation_1612_S12_0061.png
    Returns: ('ArtSpeech_Vocal_Tract_Segmentation', '1612', 'S12', '0061')
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    parts = basename.split('_')
    
    # Handle both database formats
    if 'Database' in basename:
        # ArtSpeech_Database_2_1775_S12_0001
        dataset = '_'.join(parts[:3])  # ArtSpeech_Database_2
        subject = parts[3]  # 1775
        sequence = parts[4]  # S12
        frame = parts[5]  # 0001
    else:
        # ArtSpeech_Vocal_Tract_Segmentation_1612_S12_0061
        dataset = '_'.join(parts[:4])  # ArtSpeech_Vocal_Tract_Segmentation
        subject = parts[4]  # 1612
        sequence = parts[5]  # S12
        frame = parts[6]  # 0061
    
    return dataset, subject, sequence, frame


def run_inference(config):
    """
    Run YOLO inference on test images
    
    Args:
        config: Configuration dictionary
    """
    # Load model
    model_path = config['model']['weights']
    conf_threshold = config['model']['conf_threshold']
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Load data configuration
    data_yaml_path = config['data']['data_yaml']
    data_config = load_data_yaml(data_yaml_path)
    
    # Get original image size (before YOLO resizing)
    original_size = tuple(config['data'].get('original_image_size', [136, 136]))
    
    # Build class name mapping
    class_id_to_name = data_config.get('names', {})
    class_name_mapping = config.get('class_mapping', {})
    
    # Get test image directory
    data_root = Path(data_yaml_path).parent
    split = config['data']['split']
    test_img_dir = data_root / f"images/{split}"
    
    if not test_img_dir.exists():
        # Try alternate path
        test_img_dir = data_root / data_config.get(split, f'{split}/images')
    
    if not test_img_dir.exists():
        raise ValueError(f"Test image directory not found: {test_img_dir}")
    
    # Get all test images (try both jpg and png)
    test_images = sorted(list(test_img_dir.glob('*.jpg')))
    if len(test_images) == 0:
        test_images = sorted(list(test_img_dir.glob('*.png')))
    print(f"Found {len(test_images)} test images in {test_img_dir}")
    
    # Create output directories
    base_dir = Path(config['output']['base_dir'])
    save_contours = config['output']['save_contours']
    save_masks = config['output']['save_masks']
    save_visualizations = config['output']['save_visualizations']
    
    if save_contours:
        contours_dir = base_dir / 'inference_contours'
        contours_dir.mkdir(parents=True, exist_ok=True)
        print(f"Contours will be saved to: {contours_dir}")
    
    if save_masks:
        masks_dir = base_dir / 'inference_masks'
        masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"Masks will be saved to: {masks_dir}")
    
    if save_visualizations:
        viz_dir = base_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_dir}")
    
    print(f"Confidence threshold: {conf_threshold}")
    print(f"\nProcessing images...")
    
    # Process each test image
    stats = {
        'total': 0,
        'with_predictions': 0,
        'total_contours': 0,
        'by_class': {}
    }
    
    for image_path in tqdm(test_images, desc="Running inference"):
        # Parse image path
        try:
            dataset, subject, sequence, frame = parse_image_path(str(image_path))
        except Exception as e:
            print(f"\nWarning: Could not parse image path {image_path}: {e}")
            continue
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"\nWarning: Could not load image: {image_path}")
            continue
        
        h, w = img.shape[:2]
        
        # Run inference at YOLO's native resolution (640x640 by default)
        # YOLO will automatically resize internally
        results = model.predict(
            str(image_path), 
            conf=conf_threshold,
            imgsz=640,  # Use YOLO's native 640x640 resolution for better accuracy
            verbose=False
        )[0]
        
        stats['total'] += 1
        
        # Extract predictions
        if results.masks is not None and len(results.masks) > 0:
            stats['with_predictions'] += 1
            
            # Create output directories for this sequence
            if save_contours:
                seq_contour_dir = contours_dir / dataset / subject / sequence
                seq_contour_dir.mkdir(parents=True, exist_ok=True)
            
            if save_masks:
                seq_mask_dir = masks_dir / dataset / subject / sequence
                seq_mask_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each detected instance
            pred_masks = {}
            for i, (box, mask, cls) in enumerate(zip(results.boxes, results.masks, results.boxes.cls)):
                class_id = int(cls.item())
                conf = box.conf.item()
                class_name = class_id_to_name.get(class_id, f"class_{class_id}")
                
                # Get mask as numpy array (at YOLO's inference resolution, typically 640x640)
                mask_array = mask.data[0].cpu().numpy()
                
                # Resize mask to original MRI image size (136x136)
                # This maps from YOLO's inference resolution back to the original data resolution
                mask_resized = cv2.resize(mask_array, original_size, interpolation=cv2.INTER_LINEAR)
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Store prediction (keep highest confidence if multiple)
                if class_id not in pred_masks or conf > pred_masks[class_id]['conf']:
                    pred_masks[class_id] = {
                        'mask': mask_binary,
                        'conf': conf,
                        'class_name': class_name
                    }
            
            # Save contours and/or masks for each class
            for class_id, pred_data in pred_masks.items():
                mask = pred_data['mask']
                class_name = pred_data['class_name']
                
                # Update statistics
                if class_name not in stats['by_class']:
                    stats['by_class'][class_name] = 0
                stats['by_class'][class_name] += 1
                
                # Map class name if needed (e.g., lower-lip -> lips)
                output_class_name = class_name_mapping.get(class_name, class_name)
                
                if save_contours:
                    # Extract contour
                    contour = extract_contour_from_mask(mask, class_name)
                    
                    if contour is not None:
                        # Save contour as .npy file (same format as Mask-RCNN)
                        contour_file = seq_contour_dir / f"{frame}_{output_class_name}.npy"
                        np.save(contour_file, contour.astype(np.float32))
                        stats['total_contours'] += 1
                
                if save_masks:
                    # Save full mask
                    mask_file = seq_mask_dir / f"{frame}_{output_class_name}.npy"
                    np.save(mask_file, mask)
            
            # Save visualization if requested
            if save_visualizations:
                viz_img = img.copy()
                for class_id, pred_data in pred_masks.items():
                    mask = pred_data['mask']
                    contour = extract_contour_from_mask(mask, pred_data['class_name'])
                    if contour is not None:
                        cv2.drawContours(viz_img, [contour], -1, (0, 255, 0), 2)
                
                viz_file = viz_dir / f"{dataset}_{subject}_{sequence}_{frame}.png"
                cv2.imwrite(str(viz_file), viz_img)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Total images processed: {stats['total']}")
    print(f"Images with predictions: {stats['with_predictions']}")
    
    if save_contours:
        print(f"Total contours saved: {stats['total_contours']}")
        print(f"Average contours per image: {stats['total_contours']/max(stats['total'], 1):.2f}")
    
    print(f"\nDetections by class:")
    for class_name, count in sorted(stats['by_class'].items()):
        print(f"  {class_name}: {count}")
    
    if save_contours:
        print(f"\nContours saved to: {contours_dir}")
        print(f"\nYou can now use generate_comparison_images.py to create comparison visualizations")
    
    # Save statistics
    stats_file = base_dir / 'inference_statistics.yaml'
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='YOLO unified inference script')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to inference configuration YAML file')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise ValueError(f"Configuration file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Run inference
    run_inference(config)


if __name__ == '__main__':
    main()
