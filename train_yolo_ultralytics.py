"""
YOLO Segmentation Training Script for Vocal Tract Articulators
Uses standard Ultralytics configuration format
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO Segmentation using Ultralytics config format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file (recommended)
  python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml

  # Override specific parameters
  python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml --epochs 100 --batch 8

  # Quick test run
  python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml --epochs 1 --name test_run

  # Resume training
  python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml --resume True
        """
    )
    
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='Path to Ultralytics YAML config file (e.g., config/Nam_exp_01082026/yolo_train_config.yaml)'
    )
    
    # Optional overrides
    parser.add_argument('--model', type=str, help='Override model from config')
    parser.add_argument('--data', type=str, help='Override data path from config')
    parser.add_argument('--epochs', type=int, help='Override epochs from config')
    parser.add_argument('--batch', type=int, help='Override batch size from config')
    parser.add_argument('--imgsz', type=int, help='Override image size from config')
    parser.add_argument('--device', type=str, help='Override device from config')
    parser.add_argument('--name', type=str, help='Override experiment name from config')
    parser.add_argument('--project', type=str, help='Override project directory from config')
    parser.add_argument('--resume', type=str, help='Resume training (True/False or path to checkpoint)')
    
    args = parser.parse_args()
    
    # Check if config file exists
    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.cfg}")
    
    print("="*80)
    print("YOLO SEGMENTATION TRAINING - Ultralytics Format")
    print("="*80)
    print(f"Configuration file: {args.cfg}")
    
    # Load config to get model path
    import yaml
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model from config or args
    model_path = args.model if args.model else config.get('model', 'yolov8n-seg.pt')
    print(f"Model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Prepare overrides dictionary from config file
    overrides = {k: v for k, v in config.items() if v is not None and k != 'model'}
    
    # Apply command-line overrides
    if args.model:
        overrides['model'] = args.model
    if args.data:
        overrides['data'] = args.data
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.batch:
        overrides['batch'] = args.batch
    if args.imgsz:
        overrides['imgsz'] = args.imgsz
    if args.device is not None:
        overrides['device'] = args.device
    if args.name:
        overrides['name'] = args.name
    if args.project:
        overrides['project'] = args.project
    if args.resume:
        overrides['resume'] = args.resume.lower() in ['true', '1', 'yes'] if isinstance(args.resume, str) else args.resume
    
    if overrides:
        print("\nConfiguration from file:")
        for key in ['data', 'epochs', 'batch', 'imgsz', 'device', 'name', 'project']:
            if key in overrides:
                print(f"  {key}: {overrides[key]}")
        
        if len([k for k in overrides if k in ['model', 'data', 'epochs', 'batch', 'imgsz', 'device', 'name', 'project', 'resume']]) < len(overrides):
            print(f"  ... and {len(overrides)} total parameters from config file")
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Train using the config parameters with optional overrides
    results = model.train(**overrides)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {results.save_dir / 'weights' / 'last.pt'}")
    
    return results


if __name__ == '__main__':
    main()
