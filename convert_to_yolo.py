"""
Convert Vocal Tract dataset from Mask R-CNN format to YOLO segmentation format.
YOLO segmentation format:
- Images in images/train, images/val folders
- Labels in labels/train, labels/val folders
- Each label file contains: class_id x1 y1 x2 y2 ... (normalized polygon coordinates)
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

from dataset import VocalTractMaskRCNNDataset
from helpers import load_config


def mask_to_polygon(mask):
    """
    Convert binary mask to polygon coordinates (normalized)
    Returns: list of normalized coordinates [x1, y1, x2, y2, ...]
    """
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour to reduce points
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Flatten and normalize coordinates
    h, w = mask.shape
    polygon = []
    for point in approx:
        x, y = point[0]
        # Normalize to [0, 1]
        polygon.extend([x / w, y / h])
    
    return polygon


def convert_dataset_to_yolo(config, output_dir, split='train'):
    """
    Convert dataset split to YOLO format
    
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
    
    # Load dataset
    print(f"\nLoading {split} dataset...")
    dataset = VocalTractMaskRCNNDataset(
        datadir=config.get('datadir'),
        subj_sequences=config[sequences_key],
        classes=config['classes'],
        size=tuple(config.get('size', [224, 224])),
        annotation='mask',
        mode=config.get('mode', 'gray'),
        image_folder=config.get('image_folder', 'dicoms'),
        image_ext=config.get('image_ext', 'dcm'),
        allow_missing=False,
        include_bkg=False
    )
    
    print(f"Converting {len(dataset)} images to YOLO format...")
    
    # Class mapping
    classes = sorted(config['classes'])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    converted_count = 0
    
    for idx in tqdm(range(len(dataset))):
        try:
            info, img_tensor, target_dict = dataset[idx]
            
            # Create unique filename
            subject = info['subject'].replace('/', '_')
            sequence = info['sequence']
            instance = info['instance_number']
            filename = f"{subject}_{sequence}_{instance:04d}"
            
            # Convert image tensor to PIL Image and save
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = img_np * std + mean
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            # Convert to grayscale if needed
            if config.get('mode', 'gray') == 'gray':
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                img_pil = Image.fromarray(img_np, mode='L')
            else:
                img_pil = Image.fromarray(img_np, mode='RGB')
            
            img_path = images_dir / f"{filename}.jpg"
            img_pil.save(img_path, quality=95)
            
            # Convert masks to YOLO format
            masks = target_dict['masks']
            labels = target_dict['labels']
            
            label_lines = []
            for mask, label in zip(masks, labels):
                class_name = classes[label.item()]
                class_id = class_to_idx[class_name]
                
                # Convert mask to polygon
                mask_np = mask.numpy()
                polygon = mask_to_polygon(mask_np)
                
                if polygon is None or len(polygon) < 6:  # Need at least 3 points
                    continue
                
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
            print(f"\nError processing item {idx}: {e}")
            continue
    
    print(f"Successfully converted {converted_count}/{len(dataset)} images for {split}")
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
    parser = argparse.ArgumentParser(description='Convert dataset to YOLO segmentation format')
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
    
    print(f"\nConverting dataset to YOLO format...")
    print(f"Output directory: {output_dir}")
    print(f"Classes: {config['classes']}")
    
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
