# YOLO Dataset Conversion Pipeline

## Overview

This pipeline converts ArtSpeech vocal tract segmentation data to YOLO format for instance segmentation training. The conversion preserves temporal information by encoding three consecutive frames (t-1, t, t+1) as RGB channels and converts ROI polygon annotations directly to YOLO polygon format.

## Dataset Statistics

### Final Dataset (Complete Annotations Only)
- **Total Images:** 1,130
  - Train: 789 images
  - Valid: 108 images  
  - Test: 233 images

### Breakdown by Subject

| Subject | Database | ID   | Train | Valid | Test | Total | Notes |
|---------|----------|------|-------|-------|------|-------|-------|
| S1      | ASD1     | 1612 | 71    | 9     | 20   | 100   | ✓ Complete |
| S3      | ASD1     | 1618 | 71    | 9     | 20   | 100   | ✓ Complete |
| S5      | ASD1     | 1635 | 73    | 9     | 20   | 102   | +2 extra |
| S6      | ASD1     | 1638 | 73    | 9     | 20   | 102   | +2 extra |
| S7      | ASD1     | 1640 | 74    | 9     | 20   | 103   | +3 extra |
| S7.1    | ASD1     | 1662 | 50    | 0     | 50   | 100   | ✓ No valid split |
| S7.2    | ASD2     | 1775 | 308   | 54    | 63   | 425   | ✓ ASD2 subject |
| S9      | ASD1     | 1659 | 71    | 9     | 20   | 100   | ✓ Complete |

**Notes:** 
- S2 (1617), S4 (1628), S8 (1653) excluded due to insufficient complete annotations
- S7.2 test split verified with 63 images (matches expected count exactly!)
- "Complete annotations" = all 9 articulator classes annotated in each frame

## Key Features

### 1. Temporal Information Preservation
- **R channel:** Frame t-1 (previous frame)
- **G channel:** Frame t (current frame)
- **B channel:** Frame t+1 (next frame)

### 2. Complete Annotations Filter
The conversion now includes **only frames with all 9 classes annotated**:
- arytenoid-cartilage
- epiglottis
- lower-lip
- pharynx
- soft-palate-midline
- thyroid-cartilage
- tongue
- upper-lip
- vocal-folds

This filtering ensures consistent annotation quality across all images.

### 3. Direct ROI to YOLO Conversion
- Reads ImageJ ROI polygon files directly
- Converts coordinates from original size (136×136) to target size (224×224)
- Normalizes coordinates to [0, 1] range for YOLO format
- Preserves polygon shape without quality loss from rasterization

## Usage

### 1. Convert Dataset

```bash
python convert_to_yolo.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --output data_yolo
```

**Output:**
- `data_yolo/images/{train,valid,test}/` - RGB JPEG images (224×224)
- `data_yolo/labels/{train,valid,test}/` - YOLO polygon annotations
- `data_yolo/data.yaml` - YOLO dataset configuration

### 2. Analyze Dataset Statistics

```bash
python count_dataset_by_subject.py
```

**Output:** Breakdown by subject, split, and database with total counts.

### 3. Compare with Expected Values

```bash
python compare_dataset_stats.py
```

**Output:** Comparison table showing expected vs actual image counts per subject.

### 4. Visualize Annotations

```bash
python visualize_yolo_annotations.py \
    --data_yaml ./data_yolo/data.yaml \
    --output_dir ./data_yolo/visualize_image_annotations
```

**Output:** Annotated images with colored polygons and compact legend in `data_yolo/visualize_image_annotations/`.

**Visualization Features:**
- Color-coded polygons for each articulator class
- Compact legend (130px width, top-right corner)
- Small color boxes (6px) with readable text (0.3 font scale)
- Output saved as high-quality JPEGs

### 5. Train YOLO Model

**Option A: Using config file (Recommended)**
```bash
python train_yolo_seg.py --config config/Nam_exp_01082026/yolo_seg_train.yaml
```

The config file (`yolo_seg_train.yaml`) contains all training parameters:
- `model_name`: yolov8n-seg (n=nano, s=small, m=medium, l=large, x=xlarge)
- `n_epochs`: 200
- `batch_size`: 16
- `imgsz`: 224
- `learning_rate`: 0.01
- `weight_decay`: 0.0005
- `patience`: 30 (early stopping)
- `device`: '0' (GPU) or 'cpu'

**Option B: Using command line arguments**
```bash
python train_yolo_seg.py \
    --data ./data_yolo/data.yaml \
    --model yolov8n-seg.pt \
    --epochs 200 \
    --imgsz 224 \
    --batch_size 16 \
    --device 0 \
    --lr0 0.01 \
    --weight_decay 0.0005 \
    --patience 30 \
    --project runs/yolo_seg \
    --name vocal_tract_exp
```

**Quick test run (1 epoch):**
```bash
python train_yolo_seg.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --epochs 1 \
    --name test_run
```

**Resume training:**
```bash
python train_yolo_seg.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --resume
```

**Training Output:**
- `runs/segment/vocal_tract/weights/best.pt` - Best model weights
- `runs/segment/vocal_tract/weights/last.pt` - Last checkpoint
- `runs/segment/vocal_tract/results.png` - Training curves
- `runs/segment/vocal_tract/val_batch*.jpg` - Validation visualizations

## Important Notes

### Complete Annotations Requirement
The conversion script filters out frames with incomplete annotations. For example:
- S12 frame 2900 had only tongue annotation (1/9 classes) - **excluded**
- Only frames with all 9 classes present are included - **ensures consistent quality**

### Image Format
- **Input:** 136×136 NumPy arrays (.npy files) with uint16 data
- **Processing:** 
  - Histogram normalization per frame
  - Conversion to uint8
  - Temporal RGB stacking
- **Output:** 224×224 RGB JPEG images

### Annotation Format
- **Input:** ImageJ ROI polygon files (.roi)
- **Output:** YOLO segmentation format (class_id x1 y1 x2 y2 ... xn yn)
- **Coordinates:** Normalized to [0, 1] range

## Configuration

Edit `config/Nam_exp_01082026/yolo_seg_train.yaml` to customize:
- Data paths (`datadir`)
- Train/valid/test sequence splits
- Image folder name (`image_folder: NPY_MR`)
- Image extension (`image_ext: npy`)
- Output image size (`size: [224, 224]`)

## Files

- `convert_to_yolo.py` - Main conversion script (direct ROI to YOLO)
- `visualize_yolo_annotations.py` - Visualization tool
- `count_dataset_by_subject.py` - Dataset statistics by subject analyzer
- `count_dataset_total.py` - Total dataset statistics analyzer
- `compare_dataset_stats.py` - Expected vs actual comparison(backup)

## Example Output Structure

```
data_yolo/
├── data.yaml                           # YOLO config
├── images/
│   ├── train/
│   │   ├── ArtSpeech_Vocal_Tract_Segmentation_1612_S7_0002.jpg
│   │   └── ...
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── ArtSpeech_Vocal_Tract_Segmentation_1612_S7_0002.txt
│   │   └── ...
│   ├── valid/
│   └── test/
└── visualize_image_annotations/
    ├── train_sample_0.jpg
    └── ...
```
