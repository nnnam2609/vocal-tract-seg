# Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging

## Author
**Nhat Nam Nguyen (Nam)**  
LORIA - Laboratoire Lorrain de Recherche en Informatique et ses Applications  
Université de Lorraine

> **Note:** This is a fork of the original work by Vinicius de Paulo Souza Ribeiro.  
> Original repository: [vribeiro1/vocal-tract-seg](https://github.com/vribeiro1/vocal-tract-seg)

---

## Overview

This repository implements deep learning methods for automatic segmentation of vocal tract articulators from real-time MRI sequences, exploring many segmentation approaches with Mask R-CNN, nnUNet and YOLO-based architectures.

### exp/yolo-seg Branch Focus

This branch focuses on training **YOLO11 segmentation models** for vocal tract articulator segmentation using the **ribbon mask conversion method**. The implementation provides:

- Instance segmentation of 9 vocal tract articulators
- Ribbon mask conversion for open contours
- Adaptive thickness scaling for different image sizes
- Training pipeline with Ultralytics framework
- Comprehensive evaluation and visualization tools
- Real-time inference capabilities

### Dataset

The training data comes from the **ASD1** and **ASD2** datasets, which are part of the larger multimodal MRI database of French speakers:

**Database:** [Multimodal dataset of real-time 2D and static 3D MRI of healthy French speakers](https://springernature.figshare.com/collections/Multimodal_dataset_of_real-time_2D_and_static_3D_MRI_of_healthy_French_speakers/5270387)

This comprehensive database includes:
- Real-time 2D MRI sequences of speech production
- Static 3D MRI volumes
- Multiple healthy French speakers
- Various speech tasks and phonetic contexts

For more details about the database and its applications, please refer to the [Related Publications](#related-publications) section below.

### Segmented Articulators

The system segments 9 different vocal tract structures:

1. **Arytenoid Cartilage** - Controls vocal fold positioning
2. **Epiglottis** - Guards the airway during swallowing
3. **Lower Lip** - Critical for labial consonants
4. **Pharynx** - Pharyngeal cavity shape
5. **Soft Palate Midline** - Velopharyngeal port control
6. **Thyroid Cartilage** - Laryngeal framework
7. **Tongue** - Primary articulator for most sounds
8. **Upper Lip** - Works with lower lip for bilabial sounds
9. **Vocal Folds** - Source of phonation

---

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nnnam2609/vocal-tract-seg.git
cd vocal-tract-seg
git checkout exp/yolo-seg
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
pip install ultralytics  # YOLOv8/YOLO11
pip install opencv-python numpy pyyaml tqdm pillow
pip install read-roi  # For ImageJ ROI file reading
```

3. **Install external dependencies** (optional, for tracking)

This repository requires `vt_tools` and `vt_tracker`:

```bash
# Clone the dependencies
git clone git@gitlab.inria.fr:vsouzari/vt_tools.git
git clone git@gitlab.inria.fr:vsouzari/vt_tracker.git

# Install them
pip3 install -e /path/to/vt_tools
pip3 install -e /path/to/vt_tracker
```

### Dataset Conversion

Convert ROI annotations to YOLO segmentation format:

```bash
python convert_to_yolo.py --config config/Nam_exp_01082026/yolo_seg_train.yaml
```

This converts ImageJ ROI files to YOLO segmentation format with ribbon mask conversion for open contours.

### Visualization

Visualize the converted annotations:

```bash
python visualize_yolo_annotations.py \
    --data_yaml ./data_yolo_ribbon_2_px/data.yaml \
    --output_dir ./visualizations \
    --splits test
```

### Training

Train a YOLO segmentation model using the config-based script:

```bash
python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml
```

This script reads all training parameters from the YAML config (model, epochs, batch size, learning rate, etc.) and trains the model using the Ultralytics framework.

### Inference

Run inference on test data using the config file:

```bash
python inference_yolo_with_config.py --config config/Nam_exp_01082026/inference_yolo_test.yaml
```

This loads the trained model and runs predictions on specified sequences, saving results and evaluation metrics.

---

## Features

This branch provides a comprehensive toolkit for YOLO-based vocal tract articulator segmentation with the following capabilities:

### Ribbon Mask Conversion

- **Open contour handling**: Converts polylines to thin ribbon masks (valid closed polygons)
- **Coordinate scaling**: Properly scales from original size (136x136) to target size (224x224)
- **Adaptive thickness**: Automatically adjusts ribbon width based on image size
- **Polygon closure modes**: AUTO/ALL/NONE for experimentation
- **Robust error handling**: Skips corrupted ROI files gracefully

### Training & Optimization

- Multiple YOLO model variants (YOLOv8, YOLO11: n, s, m, l, x)
- Ultralytics training framework with auto-optimization
- Configurable hyperparameters via YAML
- Mixed precision training support
- Early stopping with patience
- Model selection based on validation mAP

### Evaluation & Visualization

- **Visualization tools**: See masks, contours, and closure status
- **Multi-panel views**: Original + Contours + Masks + Composite
- **Closure markers**: Green (first point), Red (last point), Yellow (gap)
- **Statistics tracking**: Success rates per class
- **Per-articulator analysis**: Individual performance metrics

### Data Management

- Support for multiple image formats (DICOM, PNG, JPEG, NPY)
- RGB mode using temporal slices (t-1, t, t+1) as channels
- Annotation handling from ImageJ ROI files
- Dataset splitting following research protocols
- Exclusion lists for problematic frames

---

## Project Structure

```
vocal-tract-seg/
├── convert_to_yolo.py              # Main conversion script (ribbon method)
├── visualize_yolo_annotations.py   # Visualization tool
├── train_yolo_seg.py               # YOLO training script
├── inference_yolo_with_config.py   # Inference with config
├── inference_yolo_eval.py          # Inference with evaluation
├── dataset.py                      # Dataset classes
├── helpers.py                      # Utility functions
├── settings.py                     # Configuration constants
├── read_roi.py                     # ImageJ ROI file reader
│
├── config/                         # Configuration files
│   └── Nam_exp_01082026/
│       └── yolo_seg_train.yaml     # YOLO training config
│
├── data_yolo_ribbon_2_px/          # Converted YOLO dataset
│   ├── data.yaml                   # YOLO dataset config
│   ├── images/                     # Training/valid/test images
│   └── labels/                     # YOLO format labels
│
├── runs/                           # Training runs
│   └── yolo_ribbon/
│       └── exp_thickness2/
│           └── weights/
│               └── best.pt         # Best model checkpoint
│
├── visualizations/                 # Visualization outputs
│   └── test/
│       ├── *.jpg                   # 3-panel views
│       ├── contours_only/
│       ├── masks_only/
│       └── composite_only/
│
└── requirements.txt                # Python dependencies
```

---

## Configuration

### YAML Configuration File

```yaml
# config/Nam_exp_01082026/yolo_seg_train.yaml

# Dataset paths
datadir: /path/to/data
image_folder: NPY_MR
image_ext: npy

# Model configuration
model_name: yolov11x-seg  # Options: yolov8[n|s|m|l|x]-seg, yolo11[n|s|m|l|x]-seg

# Training parameters
batch_size: 16
n_epochs: 200
patience: 30
learning_rate: 0.01
weight_decay: 0.0005
size: [224, 224]  # Target image size
imgsz: 224
device: '0'

# Ribbon mask configuration
original_image_size: [136, 136]  # Original numpy array size
ribbon_thickness: 2              # Base thickness in pixels
adaptive_thickness: false        # Scale thickness with image size
output_dir: ./data_yolo_ribbon_2_px

# Polygon closure configuration
polygon_mode: auto  # Options: 'auto', 'all', 'none'
closed_articulators:
    - thyroid-cartilage
    - vocal-folds

# Class definitions
classes:
    0: arytenoid-cartilage
    1: epiglottis
    2: lower-lip
    3: pharynx
    4: soft-palate-midline
    5: thyroid-cartilage
    6: tongue
    7: upper-lip
    8: vocal-folds

# Data splits
train_sequences:
    "ArtSpeech_Vocal_Tract_Segmentation/1612": [S9, S10, S11, ...]
    "ArtSpeech_Vocal_Tract_Segmentation/1618": [S8, S9, S10, ...]
    ...
valid_sequences: {...}
test_sequences: {...}
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ribbon_thickness` | int | 5 | Base ribbon width in pixels |
| `adaptive_thickness` | bool | true | Auto-scale thickness with image size |
| `original_image_size` | list | [136, 136] | Size of numpy arrays before resizing |
| `polygon_mode` | str | 'auto' | Closure mode: 'auto', 'all', or 'none' |
| `closed_articulators` | list | See config | Which structures to close in AUTO mode |
| `model_name` | str | 'yolov11x-seg' | YOLO model variant |

---

## Ribbon Mask Conversion

### What is Ribbon Mask Conversion?

The ribbon mask method converts **open contours** (polylines where first point does not equal last point) into **thin ribbon masks** (closed polygons valid for YOLO segmentation).

```
Open Contour:         Ribbon Mask:          YOLO Polygon:
  *---*---*             ████████              ┌──────┐
                        ████████         →    └──────┘
```

### How It Works

1. **Draw thick polyline** on blank mask using `cv2.polylines(isClosed=False)`
2. **Apply morphological operations** to close gaps and smooth the ribbon
3. **Extract boundary contour** of the ribbon using `cv2.findContours()`
4. **Simplify polygon** with `cv2.approxPolyDP()` to reduce points
5. **Scale coordinates** from original size (136x136) to target size (224x224)
6. **Normalize coordinates** to [0,1] range for YOLO format

### Adaptive Thickness

When enabled, ribbon thickness automatically scales with image size:

```python
scale_factor = sqrt(target_w * target_h) / sqrt(136 * 136)
effective_thickness = base_thickness * scale_factor
```

**Example:**
- 136×136 → 224×224: thickness 2 → 3.3 pixels
- 136×136 → 640×640: thickness 2 → 9.4 pixels

**When disabled**, uses fixed thickness for all images.

### Polygon Closure Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **AUTO** | Close only `thyroid-cartilage` & `vocal-folds` | Default, matches Mask R-CNN |
| **ALL** | Close all polygons by connecting first and last points | Test if full closure helps |
| **NONE** | Keep all polygons open, use ribbon method for all | Maximum consistency |

### Command Line Usage

```bash
# Use config file settings
python convert_to_yolo.py --config config/Nam_exp_01082026/yolo_seg_train.yaml

# Override specific parameters
python convert_to_yolo.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --output_dir ./data_custom \
    --thickness 5 \
    --adaptive_thickness

# Disable adaptive thickness
python convert_to_yolo.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --no_adaptive_thickness
```

### Output Statistics

After conversion, you'll see statistics like:

```
Ribbon conversion statistics:
  tongue                : 1056 success,   24 failed (97.8%)
  thyroid-cartilage     :  945 success,    3 failed (99.7%)
  vocal-folds           :  834 success,    6 failed (99.3%)
  epiglottis            :  712 success,   12 failed (98.3%)
```
## Visualization

### Basic Visualization

```bash
python visualize_yolo_annotations.py \
    --data_yaml ./data_yolo_ribbon_2_px/data.yaml \
    --output_dir ./visualizations \
    --splits test
```

### With Closure Markers

```bash
python visualize_yolo_annotations.py \
    --data_yaml ./data_yolo_ribbon_2_px/data.yaml \
    --output_dir ./visualizations \
    --splits test \
    --show_closure
```

### Multiple Splits

```bash
python visualize_yolo_annotations.py \
    --data_yaml ./data_yolo_ribbon_2_px/data.yaml \
    --output_dir ./visualizations \
    --splits train valid test
```

### Output Structure

```
visualizations/
└── test/
    ├── subject_1640_S10_frame_001.jpg     # 3-panel view
    ├── subject_1640_S10_frame_002.jpg
    ├── contours_only/
    │   ├── subject_1640_S10_frame_001.jpg # Contours with markers
    │   └── ...
    ├── masks_only/
    │   ├── subject_1640_S10_frame_001.jpg # Filled masks
    │   └── ...
    └── composite_only/
        ├── subject_1640_S10_frame_001.jpg # All masks composite
        └── ...
```

### Closure Markers

- **Green dot**: First point of contour
- **Red dot**: Last point of contour
- **Yellow line**: Gap between first and last (for open contours)
- **Legend**: Shows `[OPEN]` or `[CLOSED]` status per class

---

## Training

### Using Config File (Recommended)

Train using the config-based script which reads all parameters from YAML:

```bash
python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml
```

**Config file includes:**
- Model selection (yolov8n-seg, yolov8s-seg, etc.)
- Training parameters (epochs, batch size, learning rate)
- Data paths and augmentation settings
- Output directories

**Example config:**
```yaml
# config/Nam_exp_01082026/yolo_train_config.yaml
model_name: yolov8n-seg
data_yaml: ./data_yolo_ribbon_2_px/data.yaml
epochs: 200
batch_size: 16
imgsz: 224
patience: 30
device: '0'
project: ./runs/yolo_ribbon
name: exp_thickness2
```

### Using Python API (Alternative)

For more control or experimentation, use the Ultralytics Python API directly:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n-seg.pt')

# Train with basic settings
results = model.train(
    data='./data_yolo_ribbon_2_px/data.yaml',
    epochs=200,
    imgsz=224,
    batch=16,
    patience=30,
    device=0
)
```

### Advanced Configuration

For advanced hyperparameter tuning, modify the config YAML or pass parameters directly:

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
results = model.train(
    data='./data_yolo_ribbon_2_px/data.yaml',
    epochs=200,
    imgsz=224,
    batch=16,
    lr0=0.01,              # Initial learning rate
    weight_decay=0.0005,   # Weight decay
    hsv_h=0.015,          # HSV augmentation
    flipud=0.0,           # Flip up-down probability
    fliplr=0.5,           # Flip left-right probability
    project='./runs/yolo_ribbon',
    name='exp_custom'
)
```

### Model Variants

| Model | Parameters | Size (MB) | Speed (ms) | mAP50-95 |
|-------|-----------|-----------|------------|----------|
| yolov8n-seg | 3.4M | 6.7 | 10-15 | Good |
| yolov8s-seg | 11.8M | 23.7 | 15-20 | Better |
| yolov8m-seg | 27.3M | 54.8 | 20-30 | Best |
| yolov8l-seg | 46.0M | 92.3 | 30-40 | Excellent |
| yolov8x-seg | 71.8M | 144.0 | 40-60 | Top |

**Recommendation:** Start with `yolov8n-seg` for quick experiments, use `yolov8s-seg` or `yolov8m-seg` for production.

### Resume Training

```python
model = YOLO('./runs/yolo_ribbon/exp_thickness2/weights/last.pt')
model.train(resume=True)
```

### Training Tips

1. **Batch size**: Start with 16, reduce if OOM
2. **Image size**: 224x224 is good for vocal tract data
3. **Patience**: 30-50 for early stopping
4. **Learning rate**: Default (0.01) usually works well
5. **Device**: Use GPU (device=0) for faster training

---

## Evaluation and Inference

### Using Config File (Recommended)

Run inference and evaluation using the config-based script:

```bash
python inference_yolo_with_config.py --config config/Nam_exp_01082026/inference_yolo_test.yaml
```

**Config file specifies:**
- Trained model path
- Test sequences
- Output directories
- Confidence threshold and IOU settings

**Example config:**
```yaml
# config/Nam_exp_01082026/inference_yolo_test.yaml
model_path: ./runs/yolo_ribbon/exp_thickness2/weights/best.pt
data_yaml: ./data_yolo_ribbon_2_px/data.yaml
test_sequences:
    "ArtSpeech_Vocal_Tract_Segmentation/1640": [S10, S11]
output_dir: ./inference_output
conf: 0.25
iou: 0.65
```

### Using Python API (Alternative)

For validation and inference with more control:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('./runs/yolo_ribbon/exp_thickness2/weights/best.pt')

# Validate
metrics = model.val()
print(f"Seg mAP50-95: {metrics.seg.map}")
print(f"Seg mAP50: {metrics.seg.map50}")

# Inference on test set
results = model.predict(
    source='./data_yolo_ribbon_2_px/images/test',
    save=True,
    conf=0.25,
    iou=0.65
)
```

### Single Image Prediction

```python
results = model.predict('path/to/image.jpg', save=True)

# Display result
import matplotlib.pyplot as plt
plt.imshow(results[0].plot())
plt.show()
```

### Extract Masks and Contours

```python
import cv2
import numpy as np

for r in results:
    # Get masks
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()  # (N, H, W)
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = r.boxes.cls.cpu().numpy()  # Class IDs
        confs = r.boxes.conf.cpu().numpy()   # Confidences
        
        # Process each mask
        for i, mask in enumerate(masks):
            # Threshold and convert to binary
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Extract contour
            contours, _ = cv2.findContours(
                mask_binary, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                print(f"Class {int(classes[i])}: {len(largest_contour)} points")
```
---

## Troubleshooting

### Low Conversion Success Rate

**Problem:**
```
tongue: 456 success, 123 failed (78.7%)  # Too low!
```

**Solutions:**
1. Increase `ribbon_thickness` in config:
   ```yaml
   ribbon_thickness: 5  # Increase from 2 to 5
   ```

2. Enable adaptive thickness:
   ```yaml
   adaptive_thickness: true
   ```

3. Check ROI files for corruption

### Ribbons Too Thin or Invisible

**Problem:** In visualization, ribbons are barely visible

**Solution:**
```bash
# Increase thickness
python convert_to_yolo.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --thickness 7
```

### Ribbons Overlapping

**Problem:** Ribbons from different objects merge together

**Solution:**
```bash
# Decrease thickness
python convert_to_yolo.py \
    --config config/Nam_exp_01082026/yolo_seg_train.yaml \
    --thickness 3
```
---

## Tips and Best Practices

### Dataset Conversion

1. **Start with adaptive thickness enabled** for consistent scaling
2. **Visualize before training** to verify ribbon quality
3. **Check success rates** - aim for greater than 95% per class
4. **Use AUTO polygon mode** as default (matches Mask R-CNN)
5. **Experiment with thickness** values: 2, 3, 5 are good starting points

### Training

1. **Start small**: Use `yolov8n-seg` for quick experiments
2. **Monitor validation**: Check mAP50 and mAP50-95 during training
3. **Use early stopping**: Set patience to 30-50 epochs
4. **Save checkpoints**: Keep best and last models
5. **Enable plots**: Set `plots=True` for training visualization

### Evaluation

1. **Visual inspection**: Always check some predictions visually
2. **Per-class analysis**: Some articulators are harder than others
3. **Confidence threshold**: Tune based on precision/recall tradeoff
4. **Multiple runs**: Test with different random seeds
5. **Compare with Mask R-CNN**: Benchmark against main branch results

### Inference

1. **Batch processing**: Process multiple images for efficiency
2. **Appropriate confidence**: 0.25 is a good default
3. **Post-processing**: Consider smoothing or filtering contours
4. **Output formats**: Save both images and labels for analysis

---

## Performance Expectations

### Conversion Statistics

**Good conversion:**
```
Ribbon conversion statistics:
  tongue                : 1056/1080 (97.8%)
  thyroid-cartilage     :  945/ 948 (99.7%)
  vocal-folds           :  834/ 840 (99.3%)
```

### Training Metrics

**Target metrics (YOLOv8n-seg, 224x224, 200 epochs):**
- Box mAP50-95: 0.65-0.75
- Box mAP50: 0.85-0.92
- Seg mAP50-95: 0.60-0.70
- Seg mAP50: 0.80-0.90

---

## Output Files and Formats

### Dataset Conversion Output

```
data_yolo_ribbon_2_px/
├── data.yaml                   # YOLO dataset config
├── images/
│   ├── train/                  # Training images (JPG)
│   ├── valid/                  # Validation images
│   └── test/                   # Test images
└── labels/
    ├── train/                  # Training labels (TXT)
    ├── valid/                  # Validation labels
    └── test/                   # Test labels
```

### YOLO Label Format

Each `.txt` file contains one line per object:
```
class_id x1 y1 x2 y2 x3 y3 ... xN yN
```

- `class_id`: Integer class index (0-8)
- `xi yi`: Normalized polygon coordinates in range [0, 1]
- Polygon is automatically closed by YOLO

**Example:**
```
6 0.521 0.456 0.523 0.458 0.525 0.460 ... 0.519 0.454
```

### Training Output

```
runs/yolo_ribbon/exp_thickness2/
├── weights/
│   ├── best.pt                 # Best model (highest mAP)
│   └── last.pt                 # Last epoch model
├── results.png                 # Training curves
├── confusion_matrix.png        # Confusion matrix
├── F1_curve.png               # F1 score curves
├── P_curve.png                # Precision curves
├── R_curve.png                # Recall curves
├── PR_curve.png               # Precision-Recall curves
├── labels.jpg                 # Training label statistics
├── labels_correlogram.jpg     # Label correlation
├── train_batch*.jpg           # Training batch samples
└── val_batch*_*.jpg           # Validation predictions
```

### Inference Output

```
runs/segment/predict/
├── *.jpg                      # Images with predictions
└── labels/
    └── *.txt                  # Predicted labels (same format as input)
```

---

## Requirements

Key dependencies (see `requirements.txt` for complete list):

```
torch>=1.7.0
torchvision>=0.8.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.19.0
scipy>=1.6.0
pyyaml>=5.4.0
tqdm>=4.60.0
pillow>=8.0.0
roifile>=2020.11.28
pandas>=1.2.0
```

External dependencies (optional):
- `vt_tools`: Vocal tract analysis utilities
- `vt_tracker`: Articulator tracking algorithms

---

## Experimental Branches

This repository contains multiple experimental branches exploring different segmentation approaches:

### Main Branch
- **Focus**: Mask R-CNN training on ASD1/ASD2 datasets
- **Status**: Production-ready with full evaluation pipeline
- **Features**: Complete training, testing, tracking, and analysis tools

### exp/yolo-seg (Current Branch)
- **Focus**: YOLO-based segmentation experiments
- **Models**: YOLOv8, YOLO11 with segmentation heads
- **Features**: Real-time segmentation, ribbon mask conversion, evaluation
- **Scripts**: YOLO training pipeline, inference, and visualization

### exp/nnnetv2
- **Focus**: nnUNet framework integration
- **Features**: Medical image segmentation, case mapping, preprocessing
- **Output**: Compatible format with Mask R-CNN for comparison

### exp/SAM2-ZERO_SHORT
- **Focus**: Segment Anything Model (SAM) 2.1 zero-shot experiments
- **Prompts**: Box prompts (23.36% IoU), point prompts (2.81% IoU)
- **Goal**: Evaluate foundation models without training

---

## Acknowledgments

This work is based on the original implementation by **Vinicius de Paulo Souza Ribeiro**.

**Original Author:**  
vinicius.ribeiro1@gmail.com  
[LinkedIn](https://www.linkedin.com/in/vribeiro1/)  
[Website](https://vribeiro1.github.io)

### Related Publications

- **Deep Supervision of the Vocal Tract Shape for Articulatory Synthesis of Speech**  
  Vinicius Ribeiro  
  Ph.D. Thesis

- **Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging**  
  Vinicius Ribeiro, Karyna Isaieva, Justine Leclere, Jacques Felblinger, Pierre-André Vuissoz, Yves Laprie  
  Nov 10, 2023 [Computer Methods and Programs in Biomedicine](https://vribeiro1.github.io/publications#:~:text=Computer%20Methods%20and%20Programs%20in%20Biomedicine)

### External Dependencies

This repository requires vt_tools ([Github](https://github.com/vribeiro1/vt_tools), [Gitlab](https://gitlab.inria.fr/vsouzari/vt_tools)) and vt_tracker ([Gitlab](https://gitlab.inria.fr/vsouzari/vt_tracker)). To install the library, follow the instructions below.

1. Clone the repos

```bash
git clone git@gitlab.inria.fr:vsouzari/vt_tools.git
git clone git@gitlab.inria.fr:vsouzari/vt_tracker.git
```

2. Install the repos

```bash
pip3 install -e /path/to/vt_tools
pip3 install -e /path/to/vt_tracker
```

---

## Quick Reference Card

```bash
# 1. Convert annotations to YOLO format
python convert_to_yolo.py --config config/Nam_exp_01082026/yolo_seg_train.yaml

# 2. Visualize converted annotations
python visualize_yolo_annotations.py \
    --data_yaml data_yolo_ribbon_2_px/data.yaml \
    --output_dir viz \
    --splits test

# 3. Train YOLO model
python train_yolo_ultralytics.py --cfg config/Nam_exp_01082026/yolo_train_config.yaml

# 4. Run inference and evaluation
python inference_yolo_with_config.py --config config/Nam_exp_01082026/inference_yolo_test.yaml
```

---

## License
To be done

---

## Contact

For questions or issues:
- Create an issue on GitHub
- Email: nhat-nam.nguyen@loria.fr

---
