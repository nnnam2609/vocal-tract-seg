#!/bin/bash
# Setup and run nnU-Net dataset preparation
# Usage: bash setup_nnunet.sh

set -e  # Exit on error

echo "=============================================="
echo "nnU-Net Setup Script for Vocal Tract Dataset"
echo "=============================================="

# Configuration
VOCAL_TRACT_DIR="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/vocal-tract-seg"
NNUNET_DIR="${VOCAL_TRACT_DIR}/nnUNet"
NNUNET_BASE="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/nnunet_data"

# Create nnU-Net data directories
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"

echo ""
echo "Creating nnU-Net data directories..."
mkdir -p "${nnUNet_raw}"
mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${nnUNet_results}"

echo "nnUNet_raw: ${nnUNet_raw}"
echo "nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "nnUNet_results: ${nnUNet_results}"

# Load Python 3.10 module
echo ""
echo "Loading Python 3.10 module..."
module load python/3.10.8_gcc-10.4.0

# Activate virtual environment
echo "Activating virtual environment..."
cd "${NNUNET_DIR}"
source venv/bin/activate

# Install required dependencies if not already installed
echo ""
echo "Checking dependencies..."
pip list | grep -q nibabel || pip install nibabel
pip list | grep -q opencv-python || pip install opencv-python
pip list | grep -q pydicom || pip install pydicom
pip list | grep -q pyyaml || pip install pyyaml
pip list | grep -q pillow || pip install pillow

# Go back to vocal-tract-seg directory
cd "${VOCAL_TRACT_DIR}"

# Run the conversion
echo ""
echo "=============================================="
echo "Converting dataset to nnU-Net format..."
echo "=============================================="

python prepare_nnunet_dataset.py \
    --config config/Nam_exp_12152025/1612_train.yaml \
    --output "${nnUNet_raw}" \
    --dataset-id 1

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Environment variables set:"
echo "  nnUNet_raw=${nnUNet_raw}"
echo "  nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "  nnUNet_results=${nnUNet_results}"
echo ""
echo "To use nnU-Net in the future, run:"
echo "  module load python/3.10.8_gcc-10.4.0"
echo "  cd ${NNUNET_DIR}"
echo "  source venv/bin/activate"
echo "  export nnUNet_raw=${nnUNet_raw}"
echo "  export nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "  export nnUNet_results=${nnUNet_results}"
echo ""
echo "Next steps:"
echo "  1. Verify dataset integrity and run preprocessing:"
echo "     nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity"
echo ""
echo "  2. Train the model (2D configuration for 2D MRI slices):"
echo "     nnUNetv2_train 1 2d 0"
echo ""
