# Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging

## Branch: feat/gen-image-and-video-comparison

This branch adds comparison image and video generation capabilities for visual evaluation of model predictions.

## Author
**Nhat Nam Nguyen (Nam)**  
LORIA - Laboratoire Lorrain de Recherche en Informatique et ses Applications  
Université de Lorraine

> **Note:** This is a fork of the original work by Vinicius de Paulo Souza Ribeiro.  
> Original repository: [vribeiro1/vocal-tract-seg](https://github.com/vribeiro1/vocal-tract-seg)

---

## New Features in This Branch

### 1. Comparison Image Generation

Generate comparison images showing ground truth vs predicted contours for visual evaluation:

**Features:**
- ✅ **Open/Closed Contour Handling**: Correctly handles open contours (tongue, pharynx, epiglottis, lips, soft-palate-midline, arytenoid-cartilage) and closed contours (thyroid-cartilage, vocal-folds)
- ✅ **Three Output Variants**:
  - `original/`: MRI images with ground truth contours only (solid lines)
  - `predict/`: MRI images with predicted contours only (dashed lines)
  - `compare/`: MRI images with both GT (solid) and predictions (dashed)
- ✅ **Color-coded Articulators**: Each articulator has a consistent color across all images
- ✅ **High Resolution**: 544x544 PNG output images

**Usage:**
```bash
cd /path/to/vocal-tract-seg

python generate_comparison_images.py --config config/generate_comparison_images.yaml
```

### YOLO Comparison (this repo)

Use the YOLO inference contours and the YOLO comparison config:

```bash
# 1) Run YOLO inference and save contours
python inference_yolo.py --config config/yolo_inference.yaml

# 2) Generate comparison images
python generate_comparison_images.py --config config/generate_comparison_images_yolo.yaml

# 3) (Optional) Generate summary video
python generate_summary_video.py --mode all \
  --comparison-dir comparison_output_yolo \
  --output-dir videos --fps 2 --model-name YOLO
```

For more details, see `README_YOLO_COMPARISON.md`.

**Configuration** (`config/generate_comparison_images.yaml`):
- Source data directory (MRI images + ground truth ROI files)
- Inference contours directory (model predictions)
- Output directory
- Test sequences to process
- Articulator classes and colors

### Required Input Format

#### 1. Ground Truth Contours
Ground truth contours must be in ImageJ ROI format (`.roi` files):

**Directory structure:**
```
{datadir}/{subject}/{sequence}/contours/{frame}_{class}.roi
```

**Example:**
```
/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/
└── ArtSpeech_Vocal_Tract_Segmentation/
    └── 1612/
        └── S7/
            ├── NPY_MR/              # MRI images (.npy files)
            └── contours/            # Ground truth ROI files
                ├── 0108_tongue.roi
                ├── 0108_lower-lip.roi
                ├── 0755_tongue.roi
                └── ...
```

**ROI file format:**
- ImageJ ROI format with x, y coordinates
- Created from manual annotations
- Each articulator in a separate file

#### 2. Model Prediction Contours (Required Format)

**IMPORTANT**: Your model must output contours in the following format:

**Directory structure:**
```
{inference_dir}/{subject}/{sequence}/{frame}_{class}.npy
```

**Example:**
```
results/32/test_outputs/inference_contours/
└── ArtSpeech_Vocal_Tract_Segmentation/
    └── 1612/
        └── S7/
            ├── 0108_tongue.npy
            ├── 0108_lower-lip.npy
            ├── 0755_tongue.npy
            └── ...
```

**NumPy array format** (`.npy` files):
```python
# Each .npy file contains a 2D numpy array of shape (N, 2)
# where N is the number of contour points
# Format: [[x1, y1], [x2, y2], ..., [xN, yN]]

import numpy as np

# Example contour for tongue with 50 points
contour = np.array([
    [45.2, 67.8],   # Point 1: (x, y)
    [46.1, 68.3],   # Point 2: (x, y)
    [47.5, 69.1],   # Point 3: (x, y)
    ...
    [44.8, 67.5]    # Point N: (x, y)
], dtype=np.float32)

# Save to .npy file
np.save('0108_tongue.npy', contour)

# When loaded:
loaded_contour = np.load('0108_tongue.npy')
print(loaded_contour.shape)  # Output: (N, 2) e.g., (50, 2)
```

**Key requirements for model output:**
- ✅ **Shape**: Must be `(N, 2)` where N ≥ 1
- ✅ **Data type**: `float32` or `float64`
- ✅ **Coordinate system**: Same as input MRI images (typically 0-135 for 136×136 images)
- ✅ **Point order**: Points should trace the contour in order (clockwise or counter-clockwise)
- ✅ **File naming**: `{frame_number}_{articulator_class}.npy`
  - Frame number: Zero-padded 4 digits (e.g., `0108`, `1234`)
  - Class name: Must match exactly one of the 9 classes (use hyphens, not underscores)

**Valid class names:**
```
arytenoid-cartilage
epiglottis
lower-lip
pharynx
soft-palate-midline
thyroid-cartilage
tongue
upper-lip
vocal-folds
```

#### 3. MRI Images

**Directory structure:**
```
{datadir}/{subject}/{sequence}/NPY_MR/{frame}.npy
```

**Format:**
- NumPy array of shape `(136, 136)` or `(H, W)`
- Data type: `uint16` or `uint8`
- Grayscale MRI image

**Example:**
```python
import numpy as np

# Load MRI image
mri = np.load('NPY_MR/0108.npy')
print(mri.shape)  # (136, 136)
print(mri.dtype)  # uint16
```

### How to Generate Inference Contours from Your Model

If you have a segmentation model (Mask R-CNN, nnUNet, YOLO, etc.), you need to:

**Step 1: Run inference and get contours**
```python
import numpy as np
from your_model import run_inference
from vt_tracker.postprocessing.calculate_contours import calculate_contour

# For each test frame
for frame_path in test_frames:
    # Run your model
    masks = run_inference(frame_path)  # Returns binary masks
    
    # For each articulator
    for class_name, mask in masks.items():
        # Extract contour from binary mask
        contour = calculate_contour(mask)  # Shape: (N, 2)
        
        # Save in required format
        output_path = f"{inference_dir}/{subject}/{sequence}/{frame_num:04d}_{class_name}.npy"
        np.save(output_path, contour)
```

**Step 2: Verify format**
```python
import numpy as np

# Check a saved contour
contour = np.load('results/32/test_outputs/inference_contours/ArtSpeech_Vocal_Tract_Segmentation/1612/S7/0108_tongue.npy')

# Verify shape
assert contour.ndim == 2, "Contour must be 2D array"
assert contour.shape[1] == 2, "Second dimension must be 2 (x, y)"
assert contour.shape[0] >= 1, "Must have at least 1 point"

print(f"✓ Contour shape: {contour.shape}")
print(f"✓ First 3 points: {contour[:3]}")
```

**Common issues and solutions:**

| Issue | Solution |
|-------|----------|
| Shape is `(2, N)` instead of `(N, 2)` | Use `contour.T` to transpose |
| Shape is `(N, 2, 1)` | Use `contour.squeeze()` to remove extra dimension |
| Points are integers | Convert to float: `contour = contour.astype(np.float32)` |
| Wrong file naming | Use `{frame:04d}_{class}.npy` format (4-digit zero-padded frame) |
| Class name with underscore | Replace with hyphen: `lower_lip` → `lower-lip` |

**Output Structure:**
```
comparison_output/
├── ArtSpeech_Vocal_Tract_Segmentation/
│   ├── 1612/
│   │   ├── S7/
│   │   │   ├── original/    # GT contours only
│   │   │   ├── predict/     # Predictions only
│   │   │   └── compare/     # Both GT + predictions
│   │   ├── S12/
│   │   └── S18/
│   └── ...
└── ArtSpeech_Database_2/
    └── 1775/
        ├── S9/
        ├── S12/
        └── ...
```

### 2. Summary Video Generation

Generate summary videos combining all comparison images into a single video for easy review:

**Features:**
- ✅ **Subject-level Organization**: Title frame for each subject showing number of sequences and frames
- ✅ **Model Name Display**: Shows model name (e.g., "MaskRCNN") in subject titles
- ✅ **Frame Labels**: Each frame shows subject/sequence/frame number
- ✅ **Configurable Playback Speed**: Adjustable FPS for comfortable viewing
- ✅ **MP4 Output**: Standard video format with H.264 encoding

**Usage:**
```bash
cd /path/to/vocal-tract-seg

# Generate video for all sequences
python generate_summary_video.py --mode all --fps 2 --model-name "MaskRCNN"

# Generate video for single sequence
python generate_summary_video.py --mode single \
    --subject "ArtSpeech_Database_2/1775" \
    --sequence "S12" \
    --fps 2
```

**Parameters:**
- `--mode`: `all` (all sequences) or `single` (one sequence)
- `--fps`: Frames per second (default: 2, lower = slower playback)
- `--model-name`: Model name to display in titles (default: "MaskRCNN")
- `--comparison-dir`: Path to comparison_output directory
- `--output-dir`: Output directory for videos

**Output:**
- Video file: `videos/all_comparisons_summary.mp4`
- Duration: ~2.2 minutes for 233 test images (at 2 fps)
- Size: ~13 MB

### 3. Key Implementation Details

**Contour Type Handling:**

Based on `VocalTractMaskRCNNDataset.closed_articulators` from `dataset.py`:

```python
# Closed articulators (endpoints connected)
CLOSED_ARTICULATORS = ['thyroid-cartilage', 'vocal-folds']

# Open articulators (endpoints NOT connected)
OPEN_ARTICULATORS = ['tongue', 'pharynx', 'epiglottis', 'lower-lip', 
                     'upper-lip', 'soft-palate-midline', 'arytenoid-cartilage']
```

**Color Scheme:**

From `vt_tools` package:
- arytenoid-cartilage: blueviolet
- epiglottis: turquoise
- lower-lip: lime
- pharynx: goldenrod
- soft-palate-midline: dodgerblue
- thyroid-cartilage: sandybrown
- tongue: darkorange
- upper-lip: magenta
- vocal-folds: hotpink

---

## Overview

This repository implements deep learning methods for automatic segmentation of vocal tract articulators from real-time MRI sequences, exploring many segmentation approaches with Mask R-CNN, nnUnet and YOLO-based models.

### Dataset

The training data comes from the **ASD1** and **ASD2** datasets, which are part of the larger multimodal MRI database of French speakers:

**Database:** [Multimodal dataset of real-time 2D and static 3D MRI of healthy French speakers](https://springernature.figshare.com/collections/Multimodal_dataset_of_real-time_2D_and_static_3D_MRI_of_healthy_French_speakers/5270387)

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

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU support)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/nnnam2609/vocal-tract-seg.git
cd vocal-tract-seg
git checkout feat/gen-image-and-video-comparison
```

2. **Install external dependencies**

This repository requires `vt_tools` and `vt_tracker`:

```bash
# Clone the dependencies
git clone git@gitlab.inria.fr:vsouzari/vt_tools.git
git clone git@gitlab.inria.fr:vsouzari/vt_tracker.git

# Install them
pip3 install -e /path/to/vt_tools
pip3 install -e /path/to/vt_tracker
```

3. **Install Python packages**
```bash
pip install -r requirements.txt
```

4. **Additional dependencies for visualization**
```bash
pip install opencv-python matplotlib pillow tqdm pyyaml
```

---

## Quick Start

### 1. Generate Comparison Images

```bash
python generate_comparison_images.py --config config/generate_comparison_images.yaml
```

This will process all test sequences and generate:
- 233 comparison images (699 total including original/predict/compare variants)
- Organized by subject and sequence

### 2. Generate Summary Video

```bash
python generate_summary_video.py --mode all --fps 2 --model-name "MaskRCNN"
```

This creates a single video (`videos/all_comparisons_summary.mp4`) showing:
- All test subjects and sequences
- Ground truth (solid lines) vs predictions (dashed lines)
- Subject titles with frame counts

---

## File Structure

```
vocal-tract-seg/
├── generate_comparison_images.py    # Generate comparison PNG images
├── generate_summary_video.py        # Generate summary MP4 video
├── config/
│   └── generate_comparison_images.yaml  # Configuration for image generation
├── comparison_output/               # Generated comparison images
│   ├── ArtSpeech_Vocal_Tract_Segmentation/
│   └── ArtSpeech_Database_2/
├── videos/                          # Generated videos
│   └── all_comparisons_summary.mp4
└── results/
    └── 32/
        └── test_outputs/
            └── inference_contours/  # Model predictions (.npy files)
```

---

## Testing

The comparison generation has been tested with:
- **233 test frames** across 43 sequences from 8 subjects
- Both ArtSpeech_Vocal_Tract_Segmentation and ArtSpeech_Database_2 datasets
- All 9 articulator classes
- Open and closed contour types

**Test Statistics:**
- Total sequences: 57 (43 with predictions)
- Total test frames: 233
- Images generated: 699 (233 × 3 variants)
- Video duration: 2.2 minutes at 2 fps

---

## Related Publications

For more information about the database and methodology, please refer to:

1. Original vocal tract segmentation work
2. ASD1/ASD2 dataset publications
3. Multimodal MRI database documentation

---

## License

This project inherits the license from the original repository.

---

## Acknowledgments

- Original implementation by Vinicius de Paulo Souza Ribeiro
- LORIA laboratory and Université de Lorraine
- Contributors to vt_tools and vt_tracker packages
- ASD1/ASD2 database creators

---

## Contact

For questions or issues specific to this branch:
- **Nam Nguyen** - [Email or contact info]

For questions about the original implementation:
- See the [original repository](https://github.com/vribeiro1/vocal-tract-seg)
