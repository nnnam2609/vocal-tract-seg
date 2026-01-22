"""
Convert ROI Contours to YOLOv8 Segmentation Labels (Ribbon Mask Approach)

This script converts ROI contour annotations (from ImageJ) into YOLOv8 segmentation format.
For open contours (polylines), it creates thin "ribbon" masks that are valid closed polygons.

Key features:
- Handles both open and closed contours
- Creates thin ribbon masks for open contours
- Properly scales coordinates from original image size (136x136) to target size (224x224)
- Supports temporal RGB format (t-1, t, t+1 frames)
- Configurable via YAML config file

Author: GitHub Copilot
Date: 2026-01-21
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from PIL import Image
from read_roi import read_roi_file


def uint16_to_uint8(img_arr, norm_hist=True):
    """Convert uint16 image to uint8 with histogram normalization"""
    max_val = np.amax(img_arr)
    img_arr = img_arr.astype(float) * 255 / max_val
    img_arr = img_arr.astype(np.uint8)
    
    if norm_hist:
        img_arr = cv2.equalizeHist(img_arr)
    
    return img_arr.astype(np.uint8)


def load_raw_image(filepath_t, filepath_t_minus_1, filepath_t_plus_1, size=(224, 224)):
    """
    Load raw temporal images (t-1, t, t+1) and combine into RGB
    
    Args:
        filepath_t: Path to current frame (npy file)
        filepath_t_minus_1: Path to previous frame (or None)
        filepath_t_plus_1: Path to next frame (or None)
        size: Target size (height, width)
    
    Returns:
        PIL RGB image with channels: R=t-1, G=t, B=t+1
    """
    # Load current frame (G channel)
    img_t = np.load(filepath_t)
    img_t = uint16_to_uint8(img_t, norm_hist=True)
    
    # Load previous frame (R channel)
    if filepath_t_minus_1 and os.path.exists(filepath_t_minus_1):
        img_t_minus_1 = np.load(filepath_t_minus_1)
        img_t_minus_1 = uint16_to_uint8(img_t_minus_1, norm_hist=True)
    else:
        img_t_minus_1 = img_t.copy()
    
    # Load next frame (B channel)
    if filepath_t_plus_1 and os.path.exists(filepath_t_plus_1):
        img_t_plus_1 = np.load(filepath_t_plus_1)
        img_t_plus_1 = uint16_to_uint8(img_t_plus_1, norm_hist=True)
    else:
        img_t_plus_1 = img_t.copy()
    
    # Stack into RGB
    rgb = np.stack([img_t_minus_1, img_t, img_t_plus_1], axis=-1)
    
    # Resize
    rgb_resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to PIL Image
    return Image.fromarray(rgb_resized, mode='RGB')


def create_ribbon_mask(contour_points, image_shape, thickness=5, dilation_iterations=0):
    """
    Create a thin ribbon mask around an open contour
    
    Args:
        contour_points: numpy array of shape (N, 2) with (x, y) pixel coordinates
        image_shape: (height, width) of the image
        thickness: thickness of the ribbon in pixels (default: 5)
        dilation_iterations: number of dilation iterations to apply (default: 0)
    
    Returns:
        Binary mask (H, W) with the ribbon filled, or None if failed
    """
    h, w = image_shape
    
    # Create empty mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert contour points to integer pixel coordinates
    points = np.array(contour_points, dtype=np.int32)
    
    # Ensure points are within image bounds
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    
    # Draw thick polyline (open contour)
    cv2.polylines(mask, [points], isClosed=False, color=1, thickness=thickness)
    
    # Optional: apply dilation to make ribbon thicker or connect gaps
    if dilation_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    # Optional: apply morphological closing to smooth the ribbon
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def extract_ribbon_polygon(mask, epsilon_factor=0.001):
    """
    Extract the outer boundary polygon from a ribbon mask
    
    Args:
        mask: Binary mask (H, W) with ribbon filled
        epsilon_factor: Simplification factor for approxPolyDP (as fraction of perimeter)
    
    Returns:
        Contour points as numpy array (N, 2), or None if extraction failed
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Check if contour is valid
    if len(contour) < 3:
        return None
    
    # Simplify polygon to reduce number of points
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    
    # Convert from (N, 1, 2) to (N, 2)
    polygon = approx.squeeze()
    
    # Ensure we have at least 3 points
    if len(polygon.shape) == 1 or len(polygon) < 3:
        return None
    
    return polygon


def polygon_to_yolo_format(polygon, image_shape):
    """
    Convert polygon points to YOLO segmentation format
    
    Args:
        polygon: numpy array (N, 2) with (x, y) pixel coordinates
        image_shape: (height, width) of the image
    
    Returns:
        List of normalized coordinates [x1, y1, x2, y2, ..., xN, yN]
    """
    h, w = image_shape
    
    # Normalize coordinates to [0, 1]
    normalized = []
    for x, y in polygon:
        x_norm = np.clip(x / w, 0.0, 1.0)
        y_norm = np.clip(y / h, 0.0, 1.0)
        normalized.extend([x_norm, y_norm])
    
    return normalized


def contour_to_yolo_ribbon(contour_points, image_shape, thickness=5, 
                           dilation_iterations=0, epsilon_factor=0.001):
    """
    Convert open contour to YOLO segmentation format using ribbon mask approach
    
    Args:
        contour_points: numpy array (N, 2) with (x, y) pixel coordinates
        image_shape: (height, width) of the image
        thickness: ribbon thickness in pixels
        dilation_iterations: number of dilation iterations
        epsilon_factor: polygon simplification factor
    
    Returns:
        List of normalized YOLO coordinates, or None if failed
    """
    # Step 1: Create ribbon mask
    ribbon_mask = create_ribbon_mask(contour_points, image_shape, thickness, dilation_iterations)
    
    if ribbon_mask is None or ribbon_mask.sum() == 0:
        return None
    
    # Step 2: Extract polygon from ribbon mask
    polygon = extract_ribbon_polygon(ribbon_mask, epsilon_factor)
    
    if polygon is None:
        return None
    
    # Step 3: Convert to YOLO format
    yolo_coords = polygon_to_yolo_format(polygon, image_shape)
    
    return yolo_coords


def roi_to_yolo_ribbon(roi_file_path, original_size, target_size, thickness=5):
    """
    Convert ROI file to YOLO ribbon polygon format
    
    Args:
        roi_file_path: Path to .roi file
        original_size: (width, height) of the original image before resizing
        target_size: (width, height) of the target image after resizing
        thickness: ribbon thickness in pixels
    
    Returns:
        List of normalized YOLO coordinates, or None if failed
    """
    try:
        # Read ROI file
        roi_dict = read_roi_file(str(roi_file_path))
        
        # Check if reading failed
        if roi_dict is None:
            return None
        
        # Get the first (and only) ROI data
        roi = list(roi_dict.values())[0]
        
        # Extract coordinates
        x_coords = roi.get('x', None)
        y_coords = roi.get('y', None)
        
        if x_coords is None or y_coords is None:
            return None
        
        x_coords = np.array(x_coords, dtype=np.float32)
        y_coords = np.array(y_coords, dtype=np.float32)
        
        if len(x_coords) < 2 or len(y_coords) < 2:
            return None
        
        # Scale coordinates from original to target size
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        x_scaled = x_coords * (target_w / orig_w)
        y_scaled = y_coords * (target_h / orig_h)
        
        # Combine into contour points
        contour_points = np.column_stack([x_scaled, y_scaled])
        
        # Convert to ribbon polygon using target image size
        yolo_coords = contour_to_yolo_ribbon(
            contour_points, (target_h, target_w),  # image_shape is (height, width)
            thickness=thickness,
            dilation_iterations=1,  # Add slight dilation to ensure connectivity
            epsilon_factor=0.002    # Moderate simplification
        )
        
        return yolo_coords
        
    except Exception as e:
        print(f"Error processing ROI file {roi_file_path}: {e}")
        return None
        print(f"Error processing ROI file {roi_file_path}: {e}")
        return None


def convert_dataset_to_yolo(config, output_dir, split, thickness=5, 
                            adaptive_thickness=True):
    """
    Convert dataset to YOLO format using ribbon mask approach
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        split: 'train', 'valid', or 'test'
        thickness: Base ribbon thickness in pixels (from config: ribbon_thickness)
        adaptive_thickness: If True, adjust thickness based on image size (from config: adaptive_thickness)
    
    Returns:
        Number of images converted
    """
    # Get configuration
    datadir = Path(config['datadir'])
    classes = config['classes']
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    sequences = config.get(f'{split}_sequences', {})
    
    image_folder = config.get('image_folder', 'dicoms')
    image_ext = config.get('image_ext', 'dcm')
    
    # Create output directories
    images_dir = output_dir / 'images' / split
    labels_dir = output_dir / 'labels' / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    ribbon_stats = {cls: {'success': 0, 'failed': 0} for cls in classes}
    
    print(f"\nProcessing {split} split...")
    
    # Iterate through sequences
    for subject, sequence_list in tqdm(sequences.items(), desc=f"Converting {split}"):
        for sequence in sequence_list:
            seq_dir = datadir / subject / sequence
            contours_dir = seq_dir / "contours"
            images_source_dir = seq_dir / image_folder
            
            if not contours_dir.exists() or not images_source_dir.exists():
                continue
            
            # Get all instance numbers from contour files
            roi_files = list(contours_dir.glob("*.roi"))
            instance_numbers = set()
            for roi_file in roi_files:
                # Extract instance number from filename: NNNN_classname.roi
                instance_num = roi_file.stem.split('_')[0]
                if instance_num.isdigit():
                    instance_numbers.add(int(instance_num))
            
            # Process each instance
            for instance_num in sorted(instance_numbers):
                # Generate output filename
                subject_name = subject.replace('/', '_')
                seq_name = sequence.replace('/', '_')
                output_name = f"{subject_name}_{seq_name}_{instance_num:04d}"
                
                # Get image paths for temporal context
                image_file_t = images_source_dir / f"{instance_num:04d}.{image_ext}"
                image_file_t_minus = images_source_dir / f"{instance_num-1:04d}.{image_ext}"
                image_file_t_plus = images_source_dir / f"{instance_num+1:04d}.{image_ext}"
                
                if not image_file_t.exists():
                    continue
                
                # Use None if temporal files don't exist
                if not image_file_t_minus.exists():
                    image_file_t_minus = None
                if not image_file_t_plus.exists():
                    image_file_t_plus = None
                
                try:
                    # Load image with temporal context
                    target_size = tuple(config.get('size', [224, 224]))
                    img_pil = load_raw_image(
                        filepath_t=str(image_file_t),
                        filepath_t_minus_1=str(image_file_t_minus) if image_file_t_minus else None,
                        filepath_t_plus_1=str(image_file_t_plus) if image_file_t_plus else None,
                        size=target_size
                    )
                    
                    # Convert to numpy for processing
                    img_rgb = np.array(img_pil)
                    h, w = img_rgb.shape[:2]
                    
                    # Adaptive thickness based on image size
                    if adaptive_thickness:
                        # Scale thickness with image size (larger images get thicker ribbons)
                        base_size = 224  # Reference size
                        scale = min(h, w) / base_size
                        adjusted_thickness = max(3, int(thickness * scale))
                    else:
                        adjusted_thickness = thickness
                    
                except Exception as e:
                    print(f"Error loading image {image_file_t}: {e}")
                    skipped_count += 1
                    continue
                
                # Process each class for this instance
                yolo_labels = []
                
                # Original image size (before resizing) - npy files are typically 136x136
                original_size = (136, 136)
                
                for class_name in classes:
                    roi_file = contours_dir / f"{instance_num:04d}_{class_name}.roi"
                    
                    if not roi_file.exists():
                        continue
                    
                    # Convert ROI to ribbon polygon
                    # Pass both original size (for ROI coords) and target size (for ribbon)
                    yolo_coords = roi_to_yolo_ribbon(
                        roi_file, 
                        original_size=original_size,
                        target_size=target_size,
                        thickness=adjusted_thickness
                    )
                    
                    if yolo_coords is not None and len(yolo_coords) >= 6:  # At least 3 points
                        class_id = class_to_id[class_name]
                        yolo_labels.append([class_id] + yolo_coords)
                        ribbon_stats[class_name]['success'] += 1
                    else:
                        ribbon_stats[class_name]['failed'] += 1
                
                # Save image and label if we have any valid annotations
                if len(yolo_labels) > 0:
                    # Save image (PIL Image)
                    output_image = images_dir / f"{output_name}.jpg"
                    img_pil.save(output_image, quality=95)
                    
                    # Save label
                    output_label = labels_dir / f"{output_name}.txt"
                    with open(output_label, 'w') as f:
                        for label in yolo_labels:
                            class_id = label[0]
                            coords = label[1:]
                            coords_str = ' '.join([f"{c:.6f}" for c in coords])
                            f.write(f"{class_id} {coords_str}\n")
                    
                    converted_count += 1
    
    # Print statistics
    print(f"\n{split.upper()} Statistics:")
    print(f"  Images converted: {converted_count}")
    print(f"  Images skipped: {skipped_count}")
    print(f"\nRibbon conversion statistics:")
    for class_name, stats in ribbon_stats.items():
        total = stats['success'] + stats['failed']
        if total > 0:
            success_rate = stats['success'] / total * 100
            print(f"  {class_name:25s}: {stats['success']:4d} success, {stats['failed']:4d} failed ({success_rate:.1f}%)")
    
    return converted_count


def create_yaml_config(output_dir, classes, ribbon_thickness):
    """
    Create YOLO data.yaml configuration file
    """
    data_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes),
        'ribbon_thickness': ribbon_thickness,
        'conversion_method': 'open_contour_ribbon_mask'
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated YOLO config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ROI contours to YOLO segmentation format using ribbon mask approach'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to dataset configuration YAML file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for YOLO dataset (default: from config)')
    parser.add_argument('--thickness', type=int, default=None,
                        help='Base ribbon thickness in pixels (default: from config or 5)')
    parser.add_argument('--adaptive_thickness', action='store_true', default=None,
                        help='Automatically adjust thickness based on image size (default: from config or True)')
    parser.add_argument('--no_adaptive_thickness', dest='adaptive_thickness', action='store_false',
                        help='Use fixed thickness for all images')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config or command line (command line overrides config)
    output_dir_path = args.output_dir if args.output_dir else config.get('output_dir', './data_yolo_ribbon')
    thickness = args.thickness if args.thickness is not None else config.get('ribbon_thickness', 5)
    adaptive_thickness = args.adaptive_thickness if args.adaptive_thickness is not None else config.get('adaptive_thickness', True)
    
    # Create output directory
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Converting ROI contours to YOLO format (Ribbon Mask Approach)")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Base ribbon thickness: {thickness} pixels")
    print(f"Adaptive thickness: {adaptive_thickness}")
    print(f"Original image size: {config.get('original_image_size', [136, 136])}")
    print(f"Target image size: {config.get('size', [224, 224])}")
    print(f"Classes: {config['classes']}")
    print(f"Image folder: {config.get('image_folder', 'dicoms')}")
    print(f"Image extension: {config.get('image_ext', 'dcm')}")
    print(f"{'='*60}\n")
    
    # Convert each split
    total_converted = 0
    for split in ['train', 'valid', 'test']:
        if f'{split}_sequences' in config:
            count = convert_dataset_to_yolo(
                config, output_dir, split,
                thickness=thickness,
                adaptive_thickness=adaptive_thickness
            )
            total_converted += count
    
    # Create YOLO config YAML
    create_yaml_config(output_dir, config['classes'], thickness)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total images converted: {total_converted}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
