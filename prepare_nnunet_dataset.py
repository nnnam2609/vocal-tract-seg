#!/usr/bin/env python
"""
Convert vocal tract dataset to nnU-Net format.

This script converts the existing vocal tract segmentation dataset 
(with separate mask files per articulator) to the nnU-Net format
(single multi-class segmentation with NIfTI files).
"""

import os
import json
import yaml
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from PIL import Image
from tqdm import tqdm

from helpers import sequences_from_dict

# Default articulator classes (from vt_tools)
CLASSES = [
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "thyroid-cartilage",
    "tongue",
    "upper-lip",
    "vocal-folds"
]


def convert_to_nnunet(
    datadir: str,
    subj_sequences: dict,
    output_dir: str,
    classes: list,
    image_folder: str = "NPY_MR",
    image_ext: str = "npy",
    split: str = "Tr",  # "Tr" for training, "Ts" for test
    case_id_offset: int = 0
):
    """
    Convert vocal tract dataset to nnU-Net format.
    
    Args:
        datadir: Root directory containing the dataset
        subj_sequences: Dictionary mapping subjects to sequences
        output_dir: Output directory (should be Dataset00X_Name folder)
        classes: List of articulator classes to include
        image_folder: Folder name containing images (e.g., "NPY_MR")
        image_ext: Image file extension (e.g., "npy", "dcm")
        split: "Tr" for training, "Ts" for test
        case_id_offset: Starting case ID number
        
    Returns:
        Number of cases processed
    """
    images_dir = os.path.join(output_dir, f"images{split}")
    labels_dir = os.path.join(output_dir, f"labels{split}")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    sequences = sequences_from_dict(datadir, subj_sequences)
    case_id = case_id_offset
    
    print(f"\nConverting {len(sequences)} sequences to nnU-Net format...")
    print(f"Split: {split}, Output: {output_dir}")
    
    for subject, sequence in tqdm(sequences, desc=f"Converting {split} sequences"):
        img_pattern = os.path.join(datadir, subject, sequence, image_folder, f"*.{image_ext}")
        images = sorted(glob(img_pattern))
        
        if len(images) == 0:
            print(f"Warning: No images found for {subject}/{sequence}")
            continue
        
        for image_filepath in images:
            image_name = os.path.basename(image_filepath).rsplit(".", 1)[0]
            masks_dir = os.path.join(datadir, subject, sequence, "masks")
            
            # Check if all masks exist for this image
            all_masks_exist = all(
                os.path.exists(os.path.join(masks_dir, f"{image_name}_{art}.png"))
                for art in classes
            )
            if not all_masks_exist:
                continue
            
            # Load image
            if image_ext == "npy":
                img_array = np.load(image_filepath)
            elif image_ext == "dcm":
                import pydicom
                dcm = pydicom.dcmread(image_filepath)
                img_array = dcm.pixel_array.astype(np.float32)
            else:
                # Try loading as regular image
                img_pil = Image.open(image_filepath).convert("L")
                img_array = np.array(img_pil, dtype=np.float32)
            
            # Normalize image to 0-1 range if needed
            if img_array.max() > 1.0:
                img_array = img_array / img_array.max()
            
            # Create combined segmentation mask (nnU-Net expects single multi-class label)
            combined_mask = np.zeros_like(img_array, dtype=np.uint8)
            
            for label_idx, art in enumerate(classes, start=1):
                mask_path = os.path.join(masks_dir, f"{image_name}_{art}.png")
                if os.path.exists(mask_path):
                    mask_img = Image.open(mask_path).convert("L")
                    mask_array = np.array(mask_img)
                    # Set pixels where mask > 0 to the class label
                    # Use > 127 threshold for binary masks
                    combined_mask[mask_array > 127] = label_idx
            
            # For 2D training, save as 2D images (H, W)
            # nnUNet 2D expects actual 2D data, not 3D with singleton dimension
            # This avoids augmentation errors with small dimensions
            img_for_save = img_array  # Shape: (H, W)
            mask_for_save = combined_mask  # Shape: (H, W)
            
            # Create case identifier
            case_name = f"case_{case_id:05d}"
            
            # Save image as NIfTI
            # _0000 suffix indicates first modality/channel
            img_nifti = nib.Nifti1Image(img_for_save.astype(np.float32), affine=np.eye(4))
            nib.save(img_nifti, os.path.join(images_dir, f"{case_name}_0000.nii.gz"))
            
            # Save label as NIfTI
            label_nifti = nib.Nifti1Image(mask_for_save.astype(np.uint8), affine=np.eye(4))
            nib.save(label_nifti, os.path.join(labels_dir, f"{case_name}.nii.gz"))
            
            case_id += 1
    
    num_cases = case_id - case_id_offset
    print(f"Converted {num_cases} cases for {split} split")
    return num_cases


def create_dataset_json(output_dir: str, classes: list, num_training: int, num_test: int = 0):
    """
    Create the dataset.json file required by nnU-Net.
    
    Args:
        output_dir: Output directory (Dataset00X_Name folder)
        classes: List of articulator class names
        num_training: Number of training cases
        num_test: Number of test cases (optional)
    """
    # Build labels dict: background (0) + articulator classes
    labels = {"background": 0}
    for idx, art in enumerate(classes, start=1):
        labels[art] = idx
    
    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        
        # Optional but recommended metadata
        "name": "VocalTractSegmentation",
        "description": "Vocal tract articulator segmentation from MRI images",
        "reference": "ArtSpeech Database",
        "licence": "",
        "release": "1.0",
        "tensorImageSize": "3D"
    }
    
    if num_test > 0:
        dataset_json["numTest"] = num_test
    
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\nCreated dataset.json with {num_training} training cases")
    print(f"Classes: {classes}")
    print(f"Saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert vocal tract dataset to nnU-Net format"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config yaml file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output base directory for nnU-Net (will create Dataset00X folder inside)"
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        help="Dataset ID number (e.g., 1 for Dataset001)"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=CLASSES,
        help="List of articulator classes to include"
    )
    parser.add_argument(
        "--test-config",
        default=None,
        help="Optional: Path to test config yaml file"
    )
    args = parser.parse_args()
    
    # Load training config
    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    
    # Create dataset folder
    dataset_name = f"Dataset{args.dataset_id:03d}_VocalTract"
    output_dir = os.path.join(args.output, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Converting Vocal Tract Dataset to nnU-Net Format")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_dir}")
    print(f"Config: {args.config}")
    
    # Convert training data (combining train + valid sequences for nnU-Net training)
    # nnU-Net will do its own train/val split during training
    print("\n" + "="*60)
    print("Converting TRAINING data (train_sequences + valid_sequences)")
    print("="*60)
    
    # Combine train and valid sequences into training set
    combined_train_sequences = train_cfg.get("train_sequences", {}).copy()
    valid_sequences = train_cfg.get("valid_sequences", {})
    
    # Merge valid_sequences into combined_train_sequences
    for key, value in valid_sequences.items():
        if key in combined_train_sequences:
            # If key exists, extend the list
            combined_train_sequences[key].extend(value)
        else:
            # If key doesn't exist, add it
            combined_train_sequences[key] = value
    
    num_training = convert_to_nnunet(
        datadir=train_cfg["datadir"],
        subj_sequences=combined_train_sequences,
        output_dir=output_dir,
        classes=args.classes,
        image_folder=train_cfg.get("image_folder", "NPY_MR"),
        image_ext=train_cfg.get("image_ext", "npy"),
        split="Tr"
    )
    
    # Convert test data from the same config or from separate test config
    num_test = 0
    test_sequences = None
    test_datadir = train_cfg["datadir"]
    
    # First, check if test_config is provided
    if args.test_config:
        with open(args.test_config) as f:
            test_cfg = yaml.safe_load(f)
        test_sequences = test_cfg.get("test_sequences", test_cfg.get("valid_sequences", {}))
        test_datadir = test_cfg["datadir"]
        print("\n" + "="*60)
        print(f"Converting TEST data from separate config: {args.test_config}")
        print("="*60)
    else:
        # Use test_sequences from the main config
        test_sequences = train_cfg.get("test_sequences", {})
        if test_sequences:
            print("\n" + "="*60)
            print("Converting TEST data (test_sequences from main config)")
            print("="*60)
    
    if test_sequences:
        num_test = convert_to_nnunet(
            datadir=test_datadir,
            subj_sequences=test_sequences,
            output_dir=output_dir,
            classes=args.classes,
            image_folder=train_cfg.get("image_folder", "NPY_MR"),
            image_ext=train_cfg.get("image_ext", "npy"),
            split="Ts",
            case_id_offset=num_training
        )
    
    # Create dataset.json
    create_dataset_json(output_dir, args.classes, num_training, num_test)
    
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"Training cases: {num_training} (train_sequences + valid_sequences)")
    if num_test > 0:
        print(f"Test cases: {num_test} (test_sequences)")
    print(f"\nNote: nnU-Net will automatically split the training data into")
    print(f"      train/validation during training using cross-validation.")
    print(f"\nNext steps:")
    print(f"1. Set environment variables:")
    print(f"   export nnUNet_raw={args.output}")
    print(f"   export nnUNet_preprocessed=/path/to/nnUNet_preprocessed")
    print(f"   export nnUNet_results=/path/to/nnUNet_results")
    print(f"2. Run preprocessing:")
    print(f"   nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"3. Train the model:")
    print(f"   nnUNetv2_train {args.dataset_id} 2d 0  # 2d since your data is 2D slices")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
