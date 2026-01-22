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


def polygon_to_mask(polygon_coords, image_shape):
    """
    Convert polygon coordinates to binary mask
    
    Args:
        polygon_coords: list of normalized coordinates [x1, y1, x2, y2, ...]
        image_shape: (height, width) of the image
    
    Returns:
        Binary mask (H, W) with polygon filled
    """
    h, w = image_shape
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(0, len(polygon_coords), 2):
        x = int(polygon_coords[i] * w)
        y = int(polygon_coords[i + 1] * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    
    return mask


def draw_polygon_on_image(image, polygon_coords, class_id, class_name, color, thickness=2, show_closure=True):
    """
    Draw a polygon contour on an image
    
    Args:
        image: numpy array (H, W, 3)
        polygon_coords: list of normalized coordinates [x1, y1, x2, y2, ...]
        class_id: class ID
        class_name: name of the class
        color: BGR color tuple
        thickness: line thickness
        show_closure: If True, mark first/last points and show gap line
    
    Returns:
        tuple: (image, is_closed, gap_distance, n_points)
    """
    h, w = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(0, len(polygon_coords), 2):
        x = int(polygon_coords[i] * w)
        y = int(polygon_coords[i + 1] * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # Check if polygon is closed
    first_point = points[0]
    last_point = points[-1]
    is_closed = np.allclose(first_point, last_point, atol=1)
    gap_distance = np.linalg.norm(first_point - last_point)
    
    # Draw only the contour line (no fill)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    
    # Optionally draw closure markers
    if show_closure and not is_closed:
        # Mark first point (green circle)
        cv2.circle(image, tuple(first_point), 3, (0, 255, 0), -1)
        # Mark last point (red X)
        cv2.circle(image, tuple(last_point), 3, (0, 0, 255), -1)
        # Draw gap line (yellow dashed)
        cv2.line(image, tuple(first_point), tuple(last_point), (0, 255, 255), 1, cv2.LINE_AA)
    
    return image, is_closed, gap_distance, len(points)


def add_legend(image, class_colors, class_names, closure_info=None):
    """
    Add a legend to the image showing class names and colors
    
    Args:
        image: numpy array (H, W, 3)
        class_colors: dict mapping class_id to BGR color
        class_names: dict mapping class_id to class name
        closure_info: dict mapping class_id to closure status ('OPEN' or 'CLOSED')
    """
    h, w = image.shape[:2]
    
    # Legend parameters - make it as compact as possible
    legend_width = 120 if closure_info else 90
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
        text = class_name
        if closure_info and class_id in closure_info:
            text += f" [{closure_info[class_id]}]"
        cv2.putText(image, text,
                   (box_x + box_size + 3, y_pos + 1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    
    return image


def visualize_yolo_annotations(data_yaml_path, output_dir, splits=['train', 'valid', 'test'], show_closure=True):
    """
    Visualize YOLO annotations for all splits
    
    Args:
        data_yaml_path: Path to YOLO data.yaml file
        output_dir: Output directory for visualizations
        splits: List of splits to visualize
        show_closure: If True, mark first/last points and show gap on open polygons
    """
    # Load data config
    data_config = load_yolo_data_yaml(data_yaml_path)
    base_path = Path(data_config['path'])
    class_names = data_config['names']
    polygon_mode = data_config.get('polygon_mode', 'unknown')
    closed_articulators = data_config.get('closed_articulators', [])
    
    print(f"Base path: {base_path}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Polygon mode: {polygon_mode}")
    if polygon_mode == 'auto':
        print(f"Closed articulators: {closed_articulators}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_visualized = 0
    total_stats = {
        'total_polygons': 0,
        'open_polygons': 0,
        'closed_polygons': 0
    }
    
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
            
            h, w = image.shape[:2]
            
            # Create separate images for different visualizations
            image_contours = image.copy()  # For contour overlay
            image_masks = image.copy()      # For mask overlay
            mask_composite = np.zeros((h, w), dtype=np.uint8)  # For mask composition
            
            # Get corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                # No annotations for this image
                continue
            
            # Read annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Track closure info for this image
            closure_info = {}
            
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
                color_bgr = (color[2], color[1], color[0])
                
                # 1. Draw contours on contour image
                image_contours, is_closed, gap, n_points = draw_polygon_on_image(
                    image_contours, polygon_coords, class_id, 
                    class_name, color_bgr, thickness=2,
                    show_closure=show_closure
                )
                
                # 2. Create mask from polygon
                mask = polygon_to_mask(polygon_coords, (h, w))
                
                # 3. Add mask to composite (each class gets unique ID)
                mask_composite[mask > 0] = class_id + 1
                
                # 4. Draw filled mask on mask image with transparency
                mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
                mask_colored[mask > 0] = color_bgr
                image_masks = cv2.addWeighted(image_masks, 1.0, mask_colored, 0.5, 0)
                
                # Track classes and closure status for legend
                classes_in_split[class_id] = color_bgr
                closure_info[class_id] = 'CLOSED' if is_closed else 'OPEN'
                
                # Update statistics
                total_stats['total_polygons'] += 1
                if is_closed:
                    total_stats['closed_polygons'] += 1
                else:
                    total_stats['open_polygons'] += 1
            
            # Add legends to images
            if classes_in_split:
                image_contours = add_legend(image_contours, classes_in_split, class_names, 
                                           closure_info if show_closure else None)
                image_masks = add_legend(image_masks, classes_in_split, class_names, 
                                        closure_info if show_closure else None)
            
            # Create a composite visualization with 3 panels
            # Top: Original with contours, Bottom left: Masks overlay, Bottom right: Mask composite
            panel_h = h // 2
            panel_w = w // 2
            
            # Resize panels
            contours_panel = cv2.resize(image_contours, (w, panel_h))
            masks_panel = cv2.resize(image_masks, (panel_w, panel_h))
            
            # Create colored composite from mask_composite
            composite_colored = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id in range(len(class_names)):
                color_bgr = COLORS[class_id % len(COLORS)]
                color_bgr = (color_bgr[2], color_bgr[1], color_bgr[0])
                composite_colored[mask_composite == class_id + 1] = color_bgr
            composite_panel = cv2.resize(composite_colored, (panel_w, panel_h))
            
            # Combine panels
            bottom_row = np.hstack([masks_panel, composite_panel])
            combined = np.vstack([contours_panel, bottom_row])
            
            # Add text labels to panels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, 'Contours + Closure Info', (10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, 'Contours + Closure Info', (10, 25), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(combined, 'Mask Overlay', (10, panel_h + 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, 'Mask Overlay', (10, panel_h + 25), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(combined, 'Mask Only', (panel_w + 10, panel_h + 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, 'Mask Only', (panel_w + 10, panel_h + 25), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Save combined visualization
            output_path_file = split_output_dir / img_path.name
            cv2.imwrite(str(output_path_file), combined)
            
            # Optionally save individual views
            output_dir_contours = split_output_dir / 'contours_only'
            output_dir_masks = split_output_dir / 'masks_only'
            output_dir_composite = split_output_dir / 'composite_only'
            output_dir_contours.mkdir(exist_ok=True)
            output_dir_masks.mkdir(exist_ok=True)
            output_dir_composite.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_dir_contours / img_path.name), image_contours)
            cv2.imwrite(str(output_dir_masks / img_path.name), image_masks)
            cv2.imwrite(str(output_dir_composite / img_path.name), composite_colored)
            
            split_count += 1
        
        print(f"Visualized {split_count} images for {split} split")
        total_visualized += split_count
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Total images visualized: {total_visualized}")
    print(f"{'='*60}")
    print(f"POLYGON STATISTICS:")
    print(f"  Total polygons: {total_stats['total_polygons']}")
    if total_stats['total_polygons'] > 0:
        print(f"  Open:   {total_stats['open_polygons']:4d} ({total_stats['open_polygons']/total_stats['total_polygons']*100:5.1f}%)")
        print(f"  Closed: {total_stats['closed_polygons']:4d} ({total_stats['closed_polygons']/total_stats['total_polygons']*100:5.1f}%)")
    print(f"{'='*60}")
    print(f"Output structure:")
    print(f"  {output_dir}/")
    print(f"    └── <split>/")
    print(f"        ├── <image>.jpg          ← Combined 3-panel view")
    print(f"        ├── contours_only/       ← Contours with closure markers")
    print(f"        ├── masks_only/          ← Filled masks overlay")
    print(f"        └── composite_only/      ← Mask composite (no image)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO segmentation annotations')
    parser.add_argument('--data_yaml', type=str, default='./data_yolo/data.yaml',
                        help='Path to YOLO data.yaml file')
    parser.add_argument('--output_dir', type=str, default='./data_yolo/visualize_image_annotations',
                        help='Output directory for visualized images')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to visualize (train, valid, test)')
    parser.add_argument('--show_closure', action='store_true', default=True,
                        help='Mark first/last points and show gap on open polygons')
    parser.add_argument('--no_show_closure', dest='show_closure', action='store_false',
                        help='Do not mark closure status')
    
    args = parser.parse_args()
    
    # Check if data.yaml exists
    if not Path(args.data_yaml).exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data_yaml}")
    
    # Visualize annotations
    visualize_yolo_annotations(args.data_yaml, args.output_dir, args.splits, args.show_closure)


if __name__ == "__main__":
    main()
