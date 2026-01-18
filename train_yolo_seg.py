"""
Training script for YOLO Segmentation on Vocal Tract dataset
Uses YOLOv8 segmentation model from Ultralytics
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import torch

from ultralytics import YOLO


def train_yolo_seg(
    data_yaml,
    model_name='yolov8n-seg.pt',
    epochs=100,
    imgsz=224,
    batch_size=16,
    device='0',
    project='runs/segment',
    name='vocal_tract',
    patience=20,
    lr0=0.01,
    weight_decay=0.0005,
    augment=True,
    resume=False,
    pretrained=True
):
    """
    Train YOLO segmentation model
    
    Args:
        data_yaml: Path to YOLO dataset YAML config
        model_name: YOLO model variant (yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        device: Device to use (cuda device id or 'cpu')
        project: Project directory
        name: Experiment name
        patience: Early stopping patience
        lr0: Initial learning rate
        weight_decay: Weight decay
        augment: Whether to use augmentation
        resume: Resume from last checkpoint
        pretrained: Use pretrained weights
    """
    
    # Check if CUDA is available
    if device != 'cpu':
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
        else:
            print(f"Using CUDA device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(int(device))}")
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Initializing YOLO model: {model_name}")
    print(f"{'='*60}")
    
    if resume:
        # Resume from checkpoint
        last_checkpoint = Path(project) / name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            print(f"Resuming from checkpoint: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            print(f"Checkpoint not found at {last_checkpoint}, starting fresh")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)
    
    # Training parameters
    print(f"\nTraining Configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Learning Rate: {lr0}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Patience: {patience}")
    print(f"  Augmentation: {augment}")
    print(f"  Pretrained: {pretrained}")
    
    # Start training
    print(f"\n{'='*60}")
    print(f"Starting Training...")
    print(f"{'='*60}\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        lr0=lr0,
        weight_decay=weight_decay,
        augment=augment,
        pretrained=pretrained,
        # Additional parameters
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        exist_ok=True,
        # Optimization
        optimizer='AdamW',
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic augmentation for last N epochs
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    
    # Print results
    print(f"\nBest Model: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"Last Model: {Path(project) / name / 'weights' / 'last.pt'}")
    
    # Validate best model
    print(f"\n{'='*60}")
    print(f"Validating Best Model...")
    print(f"{'='*60}\n")
    
    best_model = YOLO(str(Path(project) / name / 'weights' / 'best.pt'))
    metrics = best_model.val(data=data_yaml, imgsz=imgsz, device=device)
    
    print(f"\nValidation Metrics:")
    print(f"  Box mAP50: {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95: {metrics.box.map:.4f}")
    print(f"  Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLO Segmentation on Vocal Tract dataset')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to YOLO dataset YAML file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                        choices=['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 
                                'yolov8l-seg.pt', 'yolov8x-seg.pt'],
                        help='YOLO model variant')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (e.g., 0 or 0,1,2,3) or cpu')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Experiment arguments
    parser.add_argument('--project', type=str, default='runs/segment',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='vocal_tract',
                        help='Experiment name')
    
    # Options
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable augmentation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch without pretrained weights')
    
    args = parser.parse_args()
    
    # Check if data YAML exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data}")
    
    # Train model
    results, metrics = train_yolo_seg(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        augment=not args.no_augment,
        resume=args.resume,
        pretrained=not args.no_pretrained
    )
    
    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()
