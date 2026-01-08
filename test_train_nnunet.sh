#!/bin/bash
# Quick test training script (5 epochs only)
# Run this first to verify everything works before full training

set -e

echo "=============================================="
echo "nnU-Net Quick Test Training (5 epochs)"
echo "=============================================="

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: No GPU detected! Using CPU (will be slow)"
    DEVICE="cpu"
else
    echo "✓ GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda:0"
fi

# Setup environment
echo ""
echo "Setting up environment..."
module load python/3.10.8_gcc-10.4.0

NNUNET_DIR="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/vocal-tract-seg/nnUNet"
cd "${NNUNET_DIR}"
source venv/bin/activate

# Use test directories
export nnUNet_raw="/tmp/test_nnunet"
export nnUNet_preprocessed="/tmp/test_nnunet_preprocessed"
export nnUNet_results="/tmp/test_nnunet_results"

echo "Using test directories:"
echo "  nnUNet_raw: ${nnUNet_raw}"
echo "  nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "  nnUNet_results: ${nnUNet_results}"

# Check if preprocessing is done
if [ ! -d "${nnUNet_preprocessed}/Dataset001_VocalTract" ]; then
    echo ""
    echo "ERROR: Preprocessed data not found!"
    echo "Run preprocessing first:"
    echo "  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 2d"
    exit 1
fi

echo ""
echo "=============================================="
echo "Starting Quick Test Training"
echo "=============================================="
echo "Dataset: 001 (722 training cases)"
echo "Configuration: 2D U-Net"
echo "Fold: 0"
echo "Device: ${DEVICE}"
echo "Epochs: 5 (test only)"
echo ""

# Create output directory
mkdir -p "${nnUNet_results}/Dataset001_VocalTract/nnUNetTrainer__nnUNetPlans__2d/fold_0"

# Train with limited epochs for testing
if [ "$DEVICE" = "cpu" ]; then
    echo "Training on CPU (this will be very slow)..."
    CUDA_VISIBLE_DEVICES="" nnUNetv2_train 1 2d 0 --npz --c 
else
    echo "Training on GPU..."
    nnUNetv2_train 1 2d 0 --npz --c
fi

echo ""
echo "=============================================="
echo "Test Training Complete!"
echo "=============================================="
echo ""
echo "If this completed without errors, you're ready for full training!"
echo ""
echo "Next steps:"
echo "  1. Run full training: bash train_nnunet.sh"
echo "  2. Or run on permanent storage: bash setup_nnunet.sh && bash train_nnunet.sh"
echo ""
echo "Results saved to: ${nnUNet_results}/Dataset001_VocalTract/"
echo ""
