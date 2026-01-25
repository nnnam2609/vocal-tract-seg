#!/usr/bin/env python3
"""
Generate a quick summary video of all comparison images.
Uses OpenCV directly without intermediate PNG files.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def load_images_from_folder(folder):
    """Load all images from a folder."""
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((filename, img))
    
    return images


def add_text_to_frame(frame, text, position=(10, 30), font_scale=0.8, color=(255, 255, 255)):
    """Add text with background to a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    return frame


def create_title_frame(title, subtitle="", size=(800, 800)):
    """Create a title frame."""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Main title
    text_size = cv2.getTextSize(title, font, 1.5, 3)[0]
    x = (size[0] - text_size[0]) // 2
    y = size[1] // 2
    cv2.putText(frame, title, (x, y), font, 1.5, (255, 255, 255), 3)
    
    # Subtitle
    if subtitle:
        text_size = cv2.getTextSize(subtitle, font, 0.8, 2)[0]
        x = (size[0] - text_size[0]) // 2
        cv2.putText(frame, subtitle, (x, y + 50), font, 0.8, (150, 150, 150), 2)
    
    return frame


def generate_summary_video(comparison_base_dir, output_path, fps=2, target_size=(800, 800), model_name='MaskRCNN'):
    """
    Generate a summary video of all comparison images.
    
    Args:
        comparison_base_dir: Base path to comparison_output
        output_path: Path for output video
        fps: Frames per second
        target_size: Output frame size
        model_name: Name of the model to display in titles
    """
    comparison_dir = Path(comparison_base_dir)
    
    # Collect all subjects/sequences grouped by subject
    subjects_data = {}  # subject_id -> [(sequence_name, n_images), ...]
    total_image_frames = 0
    
    for dataset_dir in sorted(comparison_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for subject_dir in sorted(dataset_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = f"{dataset_dir.name}/{subject_dir.name}"
            
            for seq_dir in sorted(subject_dir.iterdir()):
                compare_path = seq_dir / "compare"
                if seq_dir.is_dir() and compare_path.exists():
                    n_images = len([f for f in os.listdir(compare_path) if f.endswith('.png')])
                    if n_images > 0:
                        if subject_id not in subjects_data:
                            subjects_data[subject_id] = []
                        subjects_data[subject_id].append((seq_dir.name, n_images))
                        total_image_frames += n_images
    
    n_subjects = len(subjects_data)
    total_sequences = sum(len(seqs) for seqs in subjects_data.values())
    total_title_frames = n_subjects * fps * 2  # 2 seconds per subject
    total_frames = total_image_frames + total_title_frames
    
    print(f"Found {n_subjects} subjects with {total_sequences} sequences")
    print(f"Total image frames: {total_image_frames}")
    print(f"Total title frames: {total_title_frames} ({n_subjects} subjects × 2s)")
    print(f"Total frames: {total_frames}")
    print(f"Estimated duration: {total_frames / fps:.1f} seconds ({total_frames / fps / 60:.1f} min)")
    
    # Create video writer using temp file with proper codec
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = str(output_path.with_suffix('.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(temp_path, fourcc, fps, target_size)
    
    if not video.isOpened():
        raise RuntimeError("Failed to create video writer")
    
    frame_count = 0
    
    # Process each subject
    for subject_id in tqdm(sorted(subjects_data.keys()), desc="Subjects"):
        subject_num = subject_id.split('/')[-1]
        sequences = subjects_data[subject_id]
        n_total_images = sum(n for _, n in sequences)
        
        # Write title frame for this subject (2 seconds)
        title = f"{model_name} - Subject {subject_num}"
        subtitle = f"{len(sequences)} sequences, {n_total_images} frames"
        title_frame = create_title_frame(title, subtitle, target_size)
        for _ in range(fps * 2):
            video.write(title_frame)
            frame_count += 1
        
        # Process all sequences for this subject
        for sequence_name, n_images in sequences:
            compare_dir = comparison_dir / subject_id / sequence_name / "compare"
            images = load_images_from_folder(str(compare_dir))
            
            for filename, img in images:
                # Resize to target size
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Add label with subject/sequence/frame
                frame_num = filename.replace('.png', '').replace('.jpg', '')
                label = f"{subject_num}/{sequence_name} | Frame: {frame_num}"
                resized = add_text_to_frame(resized, label)
                
                video.write(resized)
                frame_count += 1
    
    video.release()
    
    # Convert to MP4 using ffmpeg
    print("\nConverting to MP4...")
    final_path = str(output_path)
    os.system(f'ffmpeg -y -i "{temp_path}" -c:v libx264 -pix_fmt yuv420p -crf 18 "{final_path}" 2>/dev/null')
    
    # Remove temp file
    if os.path.exists(final_path):
        os.remove(temp_path)
        print(f"\n✓ Video saved to: {final_path}")
    else:
        print(f"\n✓ Video saved to: {temp_path} (MP4 conversion failed)")
    
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {frame_count / fps:.1f} seconds ({frame_count / fps / 60:.1f} minutes)")


def generate_single_video(subject_id, sequence_name, comparison_base_dir, output_path, fps=2, target_size=(800, 800)):
    """Generate video for a single sequence."""
    compare_dir = Path(comparison_base_dir) / subject_id / sequence_name / "compare"
    
    if not compare_dir.exists():
        raise FileNotFoundError(f"Compare directory not found: {compare_dir}")
    
    images = load_images_from_folder(str(compare_dir))
    if not images:
        raise ValueError(f"No images found in {compare_dir}")
    
    print(f"Found {len(images)} images for {subject_id}/{sequence_name}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = str(output_path.with_suffix('.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(temp_path, fourcc, fps, target_size)
    
    subject_num = subject_id.split('/')[-1]
    title = f"{subject_num}/{sequence_name}"
    
    for filename, img in tqdm(images, desc="Frames"):
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        frame_num = filename.replace('.png', '').replace('.jpg', '')
        label = f"{title} | Frame: {frame_num}"
        resized = add_text_to_frame(resized, label)
        video.write(resized)
    
    video.release()
    
    # Convert to MP4
    final_path = str(output_path)
    os.system(f'ffmpeg -y -i "{temp_path}" -c:v libx264 -pix_fmt yuv420p -crf 18 "{final_path}" 2>/dev/null')
    
    if os.path.exists(final_path):
        os.remove(temp_path)
        print(f"\n✓ Video saved to: {final_path}")
    else:
        print(f"\n✓ Video saved to: {temp_path}")
    
    print(f"  Duration: {len(images) / fps:.1f} seconds at {fps} fps")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate quick summary video')
    parser.add_argument('--mode', type=str, default='all', choices=['single', 'all'],
                        help='single = one sequence, all = all sequences')
    parser.add_argument('--subject', type=str, help='Subject ID for single mode')
    parser.add_argument('--sequence', type=str, help='Sequence name for single mode')
    parser.add_argument('--comparison-dir', type=str,
                        default='/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/vocal-tract-seg/comparison_output',
                        help='Path to comparison_output directory')
    parser.add_argument('--output-dir', type=str,
                        default='/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/vocal-tract-seg/videos',
                        help='Path to save output videos')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second')
    parser.add_argument('--model-name', type=str, default='MaskRCNN',
                        help='Model name to display in video title')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            if not args.subject or not args.sequence:
                parser.error("--subject and --sequence required for single mode")
            
            subject_num = args.subject.split('/')[-1]
            output_path = Path(args.output_dir) / f"{subject_num}_{args.sequence}.mp4"
            
            generate_single_video(args.subject, args.sequence, args.comparison_dir, str(output_path), fps=args.fps)
        
        else:  # all mode
            output_path = Path(args.output_dir) / "all_comparisons_summary.mp4"
            generate_summary_video(args.comparison_dir, str(output_path), fps=args.fps, model_name=args.model_name)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
