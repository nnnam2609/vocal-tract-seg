"""
Convert Vocal Tract dataset directly from ROI polygons to YOLO segmentation format.
This approach preserves annotation quality by avoiding mask conversion.

YOLO segmentation format:
- Images in images/train, images/val folders
- Labels in labels/train, labels/val folders
- Each label file contains: class_id x1 y1 x2 y2 ... (normalized polygon coordinates)

Image Format:
- RGB images where each channel represents temporal information:
  * R channel: frame at time t-1 (previous frame)
  * G channel: frame at time t (current frame)
  * B channel: frame at time t+1 (next frame)
- Raw images without ImageNet normalization
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

from read_roi import read_roi_file
from helpers import sequences_from_dict


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def uint16_to_uint8(img_arr, norm_hist=True):
    """
    Converts an uint16 image into a uint8 image.
    
    Args:
        img_arr (np.ndarray): uint16 image array to be converted.
        norm_hist (bool): If should perform histogram equalization.
    """
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


def roi_to_yolo_polygon(roi_file_path, original_size, target_size):
    """
    Convert ROI polygon file to YOLO format (normalized coordinates)
    
    Args:
        roi_file_path: Path to the ROI file
        original_size: (width, height) of the original image before resizing
        target_size: (width, height) of the target image after resizing
    
    Returns:
        List of normalized coordinates [x1, y1, x2, y2, ...] relative to target_size
    """
    if roi_file_path is None:
        return None
    
    try:
        # Read ROI file - it returns a dict with ROI name as key
        roi_dict = read_roi_file(roi_file_path)
        # Get the first (and only) ROI data
        roi = list(roi_dict.values())[0]
        
        x_coords = roi.get('x')
        y_coords = roi.get('y')
        
        if x_coords is None or y_coords is None or len(x_coords) < 3:
            return None
        
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        # Normalize coordinates to target size
        polygon = []
        for x, y in zip(x_coords, y_coords):
            # Scale from original to target size
            x_scaled = x * (target_w / orig_w)
            y_scaled = y * (target_h / orig_h)
            # Normalize to [0, 1] and clamp
            x_norm = max(0.0, min(1.0, x_scaled / target_w))
            y_norm = max(0.0, min(1.0, y_scaled / target_h))
            polygon.extend([x_norm, y_norm])
        
        return polygon
    except Exception as e:
        return None


def collect_data(datadir, sequences, classes, image_folder, image_ext, exclusion_list=None):
    """
    Collect all data items from the dataset
    
    Returns:
        List of dictionaries with image paths and ROI file paths
    """
    data = []
    if exclusion_list is None:
        exclusion_list = []
    
    for subject, sequence in sequences:
        seq_dir = os.path.join(datadir, subject, sequence)
        image_dir = os.path.join(seq_dir, image_folder)
        contours_dir = os.path.join(seq_dir, "contours")
        
        if not os.path.exists(image_dir):
            print(f"Warning: Image directory not found: {image_dir}")
            continue
        
        # Find all images
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(f'.{image_ext}')])
        
        for img_file in image_files:
            instance_number = int(os.path.splitext(img_file)[0])
            
            if (subject, sequence, instance_number) in exclusion_list:
                continue
            
            # Current frame
            img_path = os.path.join(image_dir, img_file)
            
            # Previous frame (t-1)
            img_m1_filename = f"{instance_number-1:04d}.{image_ext}"
            img_m1_path = os.path.join(image_dir, img_m1_filename)
            if not os.path.exists(img_m1_path):
                img_m1_path = None
            
            # Next frame (t+1)
            img_p1_filename = f"{instance_number+1:04d}.{image_ext}"
            img_p1_path = os.path.join(image_dir, img_p1_filename)
            if not os.path.exists(img_p1_path):
                img_p1_path = None
            
            # ROI files for each class
            rois = {}
            has_any_roi = False
            has_all_rois = True
            for cls in classes:
                roi_filename = f"{instance_number:04d}_{cls}.roi"
                roi_path = os.path.join(contours_dir, roi_filename)
                
                if os.path.exists(roi_path):
                    rois[cls] = roi_path
                    has_any_roi = True
                else:
                    rois[cls] = None
                    has_all_rois = False
            
            # Only include items with complete annotations (all classes present)
            if has_all_rois:
                data.append({
                    'subject': subject,
                    'sequence': sequence,
                    'instance_number': instance_number,
                    'img_path': img_path,
                    'img_m1_path': img_m1_path,
                    'img_p1_path': img_p1_path,
                    'rois': rois
                })
    
    return data


def convert_dataset_to_yolo(config, output_dir, split='train'):
    """
    Convert dataset split to YOLO format using direct ROI to polygon conversion
    
    Args:
        config: Configuration dictionary
        output_dir: Base output directory for YOLO dataset
        split: 'train', 'valid', or 'test'
    """
    # Create directories
    images_dir = Path(output_dir) / 'images' / split
    labels_dir = Path(output_dir) / 'labels' / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sequences for this split
    sequences_key = f"{split}_sequences"
    if sequences_key not in config:
        print(f"Warning: {sequences_key} not found in config")
        return 0
    
    sequences = sequences_from_dict(config['datadir'], config[sequences_key])
    
    print(f"\n{'='*60}")
    print(f"Processing {split} split...")
    print(f"{'='*60}")
    
    # Collect data
    data = collect_data(
        datadir=config['datadir'],
        sequences=sequences,
        classes=config['classes'],
        image_folder=config.get('image_folder', 'dicoms'),
        image_ext=config.get('image_ext', 'dcm'),
        exclusion_list=config.get('exclusion_list', None)
    )
    
    print(f"Found {len(data)} images")
    
    if len(data) == 0:
        return 0
    
    # Class mapping
    classes = sorted(config['classes'])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Get target size
    size = tuple(config.get('size', [224, 224]))
    
    converted_count = 0
    
    for item in tqdm(data, desc=f"Converting {split}"):
        try:
            # Create unique filename
            subject = item['subject'].replace('/', '_')
            sequence = item['sequence']
            instance = item['instance_number']
            filename = f"{subject}_{sequence}_{instance:04d}"
            
            # Load and save raw RGB image with temporal information
            img_rgb = load_raw_image(
                filepath_t=item['img_path'],
                filepath_t_minus_1=item['img_m1_path'],
                filepath_t_plus_1=item['img_p1_path'],
                size=size
            )
            
            img_path = images_dir / f"{filename}.jpg"
            img_rgb.save(img_path, quality=95)
            
            # Convert ROI polygons to YOLO format
            label_lines = []
            for cls in classes:
                roi_path = item['rois'].get(cls)
                
                if roi_path is None:
                    continue
                
                # Read and convert ROI to YOLO polygon
                # Original images are 136x136, target size is from config
                polygon = roi_to_yolo_polygon(
                    roi_file_path=roi_path,
                    original_size=(136, 136),  # Original numpy image size
                    target_size=size  # Target size after resizing
                )
                
                if polygon is None or len(polygon) < 6:  # Need at least 3 points
                    continue
                
                class_id = class_to_idx[cls]
                
                # Format: class_id x1 y1 x2 y2 ...
                line = f"{class_id}"
                for coord in polygon:
                    line += f" {coord:.6f}"
                label_lines.append(line)
            
            # Save label file
            if label_lines:
                label_path = labels_dir / f"{filename}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                converted_count += 1
            
        except Exception as e:
            print(f"\nError processing {item['img_path']}: {e}")
            continue
    
    print(f"Successfully converted {converted_count}/{len(data)} images for {split}")
    return converted_count


def create_yaml_config(output_dir, classes, splits=['train', 'valid', 'test']):
    """
    Create YOLO dataset configuration YAML file
    """
    yaml_content = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': {idx: cls for idx, cls in enumerate(sorted(classes))},
        'nc': len(classes)
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nYOLO config saved to: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to YOLO segmentation format (direct ROI to polygon)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to training config YAML file')
    parser.add_argument('--output_dir', type=str, default='./data_yolo',
                        help='Output directory for YOLO dataset')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConverting dataset to YOLO format (direct ROI to polygon)...")
    print(f"Output directory: {output_dir}")
    print(f"Classes: {config['classes']}")
    print(f"Image folder: {config.get('image_folder', 'dicoms')}")
    print(f"Image extension: {config.get('image_ext', 'dcm')}")
    
    # Convert each split
    total_converted = 0
    for split in ['train', 'valid', 'test']:
        count = convert_dataset_to_yolo(config, output_dir, split)
        total_converted += count
    
    # Create YOLO config YAML
    create_yaml_config(output_dir, config['classes'])
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total images converted: {total_converted}")
    print(f"Dataset ready for YOLO training at: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
