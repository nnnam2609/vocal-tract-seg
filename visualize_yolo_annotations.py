"""
Visualize YOLO segmentation annotations
Creates annotated images showing the polygon labels overlaid on images
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse


# Define colors for each class (RGB format)
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Light Blue
]


def load_yolo_data_yaml(data_yaml_path):
    """Load YOLO dataset configuration"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config


def draw_polygon_on_image(image, polygon_coords, class_id, class_name, color, thickness=2):
    """
    Draw a polygon contour on an image
    
    Args:
        image: numpy array (H, W, 3)
        polygon_coords: list of normalized coordinates [x1, y1, x2, y2, ...]
        class_id: class ID
        class_name: name of the class
        color: BGR color tuple
        thickness: line thickness
    """
    h, w = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(0, len(polygon_coords), 2):
        x = int(polygon_coords[i] * w)
        y = int(polygon_coords[i + 1] * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # Draw only the contour line (no fill)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    
    return image


def add_legend(image, class_colors, class_names):
    """
    Add a legend to the image showing class names and colors
    
    Args:
        image: numpy array (H, W, 3)
        class_colors: dict mapping class_id to BGR color
        class_names: dict mapping class_id to class name
    """
    h, w = image.shape[:2]
    
    # Legend parameters - make it as compact as possible
    legend_width = 90
    line_height = 10
    top_padding = 1
    bottom_padding = 1
    legend_height = top_padding + (len(class_colors) * line_height) + bottom_padding
    padding = 2
    
    # Create legend box in top-right corner
    legend_x = w - legend_width - padding
    legend_y = padding
    
    # Draw white background with black border
    cv2.rectangle(image, 
                 (legend_x, legend_y),
                 (legend_x + legend_width, legend_y + legend_height),
                 (255, 255, 255), -1)
    cv2.rectangle(image, 
                 (legend_x, legend_y),
                 (legend_x + legend_width, legend_y + legend_height),
                 (0, 0, 0), 1)
    
    # Draw each class entry
    for idx, (class_id, color) in enumerate(sorted(class_colors.items())):
        y_pos = legend_y + top_padding + idx * line_height + line_height // 2
        
        # Draw color box on the LEFT
        box_x = legend_x + 3
        box_size = 4
        cv2.rectangle(image,
                     (box_x, y_pos - box_size//2),
                     (box_x + box_size, y_pos + box_size//2),
                     color, -1)
        cv2.rectangle(image,
                     (box_x, y_pos - box_size//2),
                     (box_x + box_size, y_pos + box_size//2),
                     (0, 0, 0), 1)
        
        # Draw class name on the RIGHT of the color box
        class_name = class_names[class_id]
        cv2.putText(image, class_name,
                   (box_x + box_size + 3, y_pos + 1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    
    return image


def visualize_yolo_annotations(data_yaml_path, output_dir, splits=['train', 'valid', 'test']):
    """
    Visualize YOLO annotations for all splits
    
    Args:
        data_yaml_path: Path to YOLO data.yaml file
        output_dir: Output directory for visualizations
        splits: List of splits to visualize
    """
    # Load data config
    data_config = load_yolo_data_yaml(data_yaml_path)
    base_path = Path(data_config['path'])
    class_names = data_config['names']
    
    print(f"Base path: {base_path}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_visualized = 0
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Get paths
        images_dir = base_path / 'images' / split
        labels_dir = base_path / 'labels' / split
        split_output_dir = output_path / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {split}")
            continue
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping {split}")
            continue
        
        # Get all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"Found {len(image_files)} images")
        
        split_count = 0
        
        # Track which classes appear in this split for the legend
        classes_in_split = {}
        
        for img_path in tqdm(image_files, desc=f"Visualizing {split}"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Get corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                # No annotations for this image
                continue
            
            # Read annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw each annotation
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:  # Need at least class_id + 3 points (6 coordinates)
                    continue
                
                class_id = int(parts[0])
                polygon_coords = [float(x) for x in parts[1:]]
                
                # Get class name and color
                class_name = class_names[class_id]
                color = COLORS[class_id % len(COLORS)]
                
                # Draw polygon contour (OpenCV uses BGR format)
                color_bgr = (color[2], color[1], color[0])
                image = draw_polygon_on_image(image, polygon_coords, class_id, 
                                             class_name, color_bgr, thickness=2)
                
                # Track classes for legend
                classes_in_split[class_id] = color_bgr
            
            # Add legend to the image
            if classes_in_split:
                image = add_legend(image, classes_in_split, class_names)
            
            # Save visualized image
            output_path_file = split_output_dir / img_path.name
            cv2.imwrite(str(output_path_file), image)
            split_count += 1
        
        print(f"Visualized {split_count} images for {split} split")
        total_visualized += split_count
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Total images visualized: {total_visualized}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO segmentation annotations')
    parser.add_argument('--data_yaml', type=str, default='./data_yolo/data.yaml',
                        help='Path to YOLO data.yaml file')
    parser.add_argument('--output_dir', type=str, default='./data_yolo/visualize_image_annotations',
                        help='Output directory for visualized images')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to visualize (train, valid, test)')
    
    args = parser.parse_args()
    
    # Check if data.yaml exists
    if not Path(args.data_yaml).exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data_yaml}")
    
    # Visualize annotations
    visualize_yolo_annotations(args.data_yaml, args.output_dir, args.splits)


if __name__ == "__main__":
    main()
