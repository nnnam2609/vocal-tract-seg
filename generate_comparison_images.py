#!/usr/bin/env python3
"""
Generate comparison images showing ground truth vs predicted contours for plot alignment tool.
"""

import argparse
import os
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from read_roi import read_roi_file, read_roi_zip

# Color scheme from vt_tools
ARTICULATOR_COLORS = {
    'arytenoid-cartilage': '#8A2BE2',
    'epiglottis': '#FF1493',
    'incisor': '#00BFFF',
    'lower-lip': '#32CD32',
    'palate': '#FFD700',
    'pharynx': '#FF6347',
    'tongue': '#FF69B4',
    'upper-lip': '#00CED1',
    'vocal-folds': '#9370DB'
}

# Articulators with closed contours (from dataset.py VocalTractMaskRCNNDataset.closed_articulators)
# All other articulators have open contours
CLOSED_ARTICULATORS = ['thyroid-cartilage', 'vocal-folds']

def load_mri_image(mri_path):
    """Load MRI image from .npy file."""
    if not os.path.exists(mri_path):
        return None
    return np.load(mri_path)

def load_ground_truth_contour(roi_path):
    """Load ground truth contour from individual ROI file."""
    roi_path_str = str(roi_path)
    if not os.path.exists(roi_path_str):
        return None
    
    try:
        roi_dict = read_roi_file(roi_path_str)
        # ROI file has structure: {roi_name: {x: [...], y: [...], ...}}
        # Get the first (and only) ROI in the file
        for roi_name, roi_data in roi_dict.items():
            if 'x' in roi_data and 'y' in roi_data:
                x_coords = roi_data['x']
                y_coords = roi_data['y']
                return np.column_stack([x_coords, y_coords])
    except Exception as e:
        pass
    
    return None

def load_predicted_contour(pred_path):
    """Load predicted contour from .npy file."""
    if not os.path.exists(pred_path):
        return None
    
    contour = np.load(pred_path)
    if contour.shape[0] == 0:
        return None
    
    return contour

def create_image_with_contours(mri_image, contours_data, output_path, show_legend=False, dpi=100):
    """
    Create image with contours overlaid using matplotlib.
    
    Args:
        mri_image: numpy array of MRI image
        contours_data: list of tuples (contour, color, label, linestyle, articulator_class)
        output_path: path to save output image
        show_legend: whether to show legend
        dpi: dots per inch for output image
    """
    # Create figure with specific size to get desired resolution
    # For 544x544 output at 100 dpi, we need 5.44x5.44 inch figure
    fig_size = 544 / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    
    # Display MRI image
    ax.imshow(mri_image, cmap='gray', origin='upper')
    
    # Draw contours
    for contour, color, label, linestyle, articulator_class in contours_data:
        if contour is not None and len(contour) > 0:
            # Only close contour for closed articulators (thyroid-cartilage, vocal-folds)
            if articulator_class in CLOSED_ARTICULATORS:
                plot_contour = np.vstack([contour, contour[0]])
            else:
                plot_contour = contour
            ax.plot(plot_contour[:, 0], plot_contour[:, 1], 
                   color=color, linewidth=1.5, linestyle=linestyle, label=label)
    
    # Remove axes
    ax.axis('off')
    
    # Add legend if requested
    if show_legend and any(c[0] is not None for c in contours_data):
        ax.legend(loc='upper right', fontsize=5, framealpha=0.7)
    
    # Save with tight layout to remove whitespace
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def get_available_frames(datadir, inference_dir, subject_id, sequence_name):
    """Get list of available frame numbers that have inference contours."""
    # Check inference directory for available frames
    inf_dir = inference_dir / subject_id / sequence_name
    if not inf_dir.exists():
        return []
    
    frames = set()
    for npy_file in inf_dir.glob("*.npy"):
        # Extract frame number from filename like: 0061_tongue.npy
        parts = npy_file.stem.split('_')
        if len(parts) >= 2:
            try:
                frame_num = int(parts[0])
                frames.add(frame_num)
            except ValueError:
                continue
    
    return sorted(list(frames))

def process_sequence(config, subject_id, sequence_name):
    """Process one sequence and generate comparison images."""
    print(f"\nProcessing {subject_id}/{sequence_name}...")
    
    # Create output directories
    output_base = Path(config['output_dir']) / subject_id / sequence_name
    original_dir = output_base / 'original'
    predict_dir = output_base / 'predict'
    compare_dir = output_base / 'compare'
    
    for dir_path in [original_dir, predict_dir, compare_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    datadir = Path(config['datadir'])
    inference_dir = Path(config['inference_dir'])
    classes = config['classes']
    
    # Get frames that have inference contours
    frames = get_available_frames(datadir, inference_dir, subject_id, sequence_name)
    if not frames:
        print(f"  Warning: No inference frames found for {subject_id}/{sequence_name}")
        return
    
    for frame_num in tqdm(frames, desc=f"{subject_id}/{sequence_name}"):
        frame_str = str(frame_num).zfill(4)
        
        # Load MRI image - filename is just frame number
        mri_filename = f"{frame_str}.npy"
        mri_path = datadir / subject_id / sequence_name / "NPY_MR" / mri_filename
        mri_image = load_mri_image(mri_path)
        
        if mri_image is None:
            print(f"  Warning: MRI image not found: {mri_path}")
            continue
        
        # Prepare contours for each variant
        gt_contours = []
        pred_contours = []
        all_contours = []
        
        for class_name in classes:
            color = ARTICULATOR_COLORS.get(class_name, '#FFFFFF')
            
            # Load ground truth - individual ROI file
            roi_filename = f"{frame_str}_{class_name}.roi"
            roi_path = datadir / subject_id / sequence_name / "contours" / roi_filename
            gt_contour = load_ground_truth_contour(roi_path)
            
            # Load prediction - organized in subdirectories
            pred_filename = f"{frame_str}_{class_name}.npy"
            pred_path = inference_dir / subject_id / sequence_name / pred_filename
            pred_contour = load_predicted_contour(pred_path)
            
            # Add to lists (only if contour exists)
            if gt_contour is not None:
                gt_contours.append((gt_contour, color, f"{class_name} (GT)", '-', class_name))
                all_contours.append((gt_contour, color, f"{class_name} (GT)", '-', class_name))
            
            if pred_contour is not None:
                pred_contours.append((pred_contour, color, f"{class_name} (Pred)", '--', class_name))
                all_contours.append((pred_contour, color, f"{class_name} (Pred)", '--', class_name))
        
        # Generate three variants
        # Use just the frame number for output filename
        output_filename = f"{frame_str}.png"
        
        # 1. Original (GT only)
        create_image_with_contours(mri_image, gt_contours, 
                                   original_dir / output_filename, 
                                   show_legend=False)
        
        # 2. Predict (predictions only)
        create_image_with_contours(mri_image, pred_contours, 
                                   predict_dir / output_filename, 
                                   show_legend=False)
        
        # 3. Compare (both with legend)
        create_image_with_contours(mri_image, all_contours, 
                                   compare_dir / output_filename, 
                                   show_legend=True)

def main():
    parser = argparse.ArgumentParser(description='Generate comparison images for plot alignment')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Output directory: {config['output_dir']}")
    
    # Process each subject and sequence
    total_sequences = sum(len(sequences) for sequences in config['test_sequences'].values())
    print(f"\nTotal sequences to process: {total_sequences}")
    
    for subject_id, sequence_list in config['test_sequences'].items():
        for sequence_name in sequence_list:
            process_sequence(config, subject_id, sequence_name)
    
    print("\n✓ All comparison images generated successfully!")
    print(f"Output location: {config['output_dir']}")

if __name__ == '__main__':
    main()
