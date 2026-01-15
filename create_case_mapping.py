#!/usr/bin/env python
"""
Create a mapping file between nnUNet case numbers and original subject/sequence/frame.
This allows us to add subject, sequence, frame columns to evaluation results.
"""

import os
import json
import yaml
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm

from helpers import sequences_from_dict


def create_case_mapping(
    datadir: str,
    subj_sequences: dict,
    image_folder: str = "NPY_MR",
    image_ext: str = "npy",
    case_id_offset: int = 0,
    classes: list = None
):
    """
    Create mapping between case IDs and original subject/sequence/frame.
    
    Args:
        datadir: Root directory containing the dataset
        subj_sequences: Dictionary mapping subjects to sequences
        image_folder: Folder name containing images
        image_ext: Image file extension
        case_id_offset: Starting case ID number
        classes: List of articulator classes (to check if masks exist)
        
    Returns:
        List of dictionaries with case_id, subject, sequence, frame mappings
    """
    sequences = sequences_from_dict(datadir, subj_sequences)
    case_id = case_id_offset
    mapping = []
    
    print(f"Creating case mapping for {len(sequences)} sequences...")
    
    for subject, sequence in tqdm(sequences, desc="Building mapping"):
        img_pattern = os.path.join(datadir, subject, sequence, image_folder, f"*.{image_ext}")
        images = sorted(glob(img_pattern))
        
        if len(images) == 0:
            print(f"Warning: No images found for {subject}/{sequence}")
            continue
        
        for image_filepath in images:
            image_name = os.path.basename(image_filepath).rsplit(".", 1)[0]
            
            # If classes are provided, check if all masks exist
            if classes:
                masks_dir = os.path.join(datadir, subject, sequence, "masks")
                all_masks_exist = all(
                    os.path.exists(os.path.join(masks_dir, f"{image_name}_{art}.png"))
                    for art in classes
                )
                if not all_masks_exist:
                    continue
            
            # Extract frame number from image name
            # Typically: XXXX.npy or XXXX.dcm where XXXX is frame number
            try:
                frame = int(image_name)
            except ValueError:
                # If name is not a pure number, use the full name
                frame = image_name
            
            mapping.append({
                "case_id": case_id,
                "case_name": f"case_{case_id:05d}",
                "subject": subject.split("/")[-1],  # Extract just the subject ID (e.g., 1612)
                "sequence": sequence,
                "frame": frame,
                "original_path": image_filepath
            })
            
            case_id += 1
    
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Create case-to-metadata mapping for nnUNet dataset"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config yaml file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path for the mapping"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "arytenoid-cartilage",
            "epiglottis",
            "lower-lip",
            "pharynx",
            "soft-palate-midline",
            "thyroid-cartilage",
            "tongue",
            "upper-lip",
            "vocal-folds"
        ],
        help="List of articulator classes"
    )
    parser.add_argument(
        "--test-config",
        default=None,
        help="Optional: Path to test config yaml file"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to create mapping for"
    )
    args = parser.parse_args()
    
    # Load training config
    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    
    all_mappings = []
    
    # Training data mapping
    if args.split in ["train", "both"]:
        print("\n" + "="*60)
        print("Creating mapping for TRAINING data")
        print("="*60)
        
        # Combine train and valid sequences
        combined_train_sequences = train_cfg.get("train_sequences", {}).copy()
        valid_sequences = train_cfg.get("valid_sequences", {})
        
        for key, value in valid_sequences.items():
            if key in combined_train_sequences:
                combined_train_sequences[key].extend(value)
            else:
                combined_train_sequences[key] = value
        
        train_mapping = create_case_mapping(
            datadir=train_cfg["datadir"],
            subj_sequences=combined_train_sequences,
            image_folder=train_cfg.get("image_folder", "NPY_MR"),
            image_ext=train_cfg.get("image_ext", "npy"),
            case_id_offset=0,
            classes=args.classes
        )
        
        for item in train_mapping:
            item["split"] = "train"
        
        all_mappings.extend(train_mapping)
        print(f"Created {len(train_mapping)} training case mappings")
    
    # Test data mapping
    if args.split in ["test", "both"]:
        print("\n" + "="*60)
        print("Creating mapping for TEST data")
        print("="*60)
        
        test_sequences = None
        test_datadir = train_cfg["datadir"]
        
        if args.test_config:
            with open(args.test_config) as f:
                test_cfg = yaml.safe_load(f)
            test_sequences = test_cfg.get("test_sequences", test_cfg.get("valid_sequences", {}))
            test_datadir = test_cfg["datadir"]
        else:
            test_sequences = train_cfg.get("test_sequences", {})
        
        if test_sequences:
            # Calculate offset (number of training cases)
            train_offset = len(all_mappings) if args.split == "both" else 0
            
            test_mapping = create_case_mapping(
                datadir=test_datadir,
                subj_sequences=test_sequences,
                image_folder=train_cfg.get("image_folder", "NPY_MR"),
                image_ext=train_cfg.get("image_ext", "npy"),
                case_id_offset=train_offset,
                classes=args.classes
            )
            
            for item in test_mapping:
                item["split"] = "test"
            
            all_mappings.extend(test_mapping)
            print(f"Created {len(test_mapping)} test case mappings")
    
    # Save to CSV
    if all_mappings:
        df = pd.DataFrame(all_mappings)
        df.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print(f"Mapping saved to: {args.output}")
        print(f"Total cases: {len(df)}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few entries:")
        print(df.head(10))
        print(f"{'='*60}\n")
    else:
        print("\nNo mappings created!")


if __name__ == "__main__":
    main()
