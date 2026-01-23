# Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging

## Author
**Nhat Nam Nguyen (Nam)**  
LORIA - Laboratoire Lorrain de Recherche en Informatique et ses Applications  
Université de Lorraine

> **Note:** This is a fork of the original work by Vinicius de Paulo Souza Ribeiro.  
> Original repository: [vribeiro1/vocal-tract-seg](https://github.com/vribeiro1/vocal-tract-seg)

---

## Overview

This repository implements deep learning methods for automatic segmentation of vocal tract articulators from real-time MRI sequences, exploring many segmentation approaches with Mask R-CNN, nnUnet and YOLO-based .

## gen-video Branch (Model Comparison Videos)

This branch adds a lightweight tool to generate **side-by-side videos** for different
models using the **same test frames**. It is designed for Mask R-CNN, YOLO, nnUNet,
and MedSAM outputs as long as the frame names are consistent.

Tool: `tools/gen_video.py`

### Features
- Uses a frame list (one filename per line) to keep all models aligned
- Flexible model input format with filename templates
- Optional raw input column
- Optional audio merge (moviepy) and silence removal (TextGrid)

### Quick Example (YOLO vs MedSAM)

```bash
python tools/gen_video.py \
  --frames config/Nam_exp_01082026/test_image_list_233.txt \
  --input-dir data_yolo_ribbon_2_px/images/test \
  --model yolo=./inference_output_yolo_Nam_exp_01232026_all1:eval_{stem}.png \
  --model medsam=./inference_output_medsam2_Nam_exp_01232026_all1:eval_{stem}.png \
  --output ./videos/yolo_vs_medsam.mp4 \
  --fps 50.05 \
  --tile-width 640 --tile-height 480
```

### Template Rules

`--model name=DIR[:TEMPLATE]`
- `{name}` = full filename (e.g., `ArtSpeech_Database_2_1775_S12_0001.jpg`)
- `{stem}` = filename without extension (e.g., `ArtSpeech_Database_2_1775_S12_0001`)
- If TEMPLATE is omitted, default is `eval_{stem}.png`

### Optional Audio Merge

```bash
python tools/gen_video.py ... \
  --audio /path/to/1775_S12.wav \
  --with-audio
```

To remove silence using a TextGrid:
```bash
python tools/gen_video.py ... \
  --audio /path/to/1775_S12.wav \
  --with-audio \
  --no-silence /path/to/1775_S12_adjusted.textgrid
```

### Main Branch Focus

This branch focuses on training **Mask R-CNN** models for vocal tract articulator segmentation using data from the **ASD1** and **ASD2** datasets. The implementation provides:

- Instance segmentation of 9 vocal tract articulators
- Training pipeline with data augmentation
- Comprehensive evaluation metrics (P2CP, Jaccard Index)
- Articulator tracking across temporal sequences
- Multiple output formats for research analysis

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

3. **Install project requirements**
```bash
pip install -r requirements.txt
```

4. **Activate the environment** (if using a specific virtual environment)
```bash
source /path/to/your/env/bin/activate
```

### Training

To train a Mask R-CNN model:

```bash
# Activate environment (example path)
source /path/to/your/env/bin/activate

# Run training with your configuration file
python train_maskrcnn.py with /path/to/your/config/thesis_config/your config.yaml
```

**Configuration files** for different subjects and experiments can be found in:
- `config/thesis_config/` - Training configurations for different subjects

### Inference

To run inference on new data:

```bash
# Activate environment
source /path/to/your/env/bin/activate

# Run inference
python inference_maskrcnn.py with /path/to/your/inference_config.yaml
```

### Testing/Evaluation

To evaluate a trained model:

```bash
python test_maskrcnn.py with /path/to/your/test_config.yaml
```

This will generate evaluation metrics including P2CP (Point-to-Curve-Point) distances and Jaccard Index for each articulator.

---

## Features

This repository provides a comprehensive toolkit for vocal tract articulator segmentation with the following capabilities:

### Multi-Model Support
- **Mask R-CNN**: Instance segmentation with ResNet50-FPN backbone
- **nnUNet**: Medical image segmentation framework integration
- **YOLO**: Real-time instance segmentation industial preference
- **SAM 2.1**: Zero-shot segmentation experiments with foundation models
- **DeepLabV3**: Alternative semantic segmentation architecture

### Training & Optimization
- Multi-stage training pipeline with Sacred experiment tracking
- Multiple learning rate schedulers (CyclicLR, ReduceOnPlateau)
- Weight decay and regularization techniques
- Mixed precision training support
- Model parameter counting and architecture analysis
- Automatic best model selection based on validation metrics

### Evaluation & Metrics
- **P2CP (Point-to-Curve-Point)**: Distance metrics (RMS, Mean, Std)
- **Jaccard Index**: Overlap similarity for closed contours
- **Per-articulator statistics**: Individual performance analysis
- **Temporal consistency**: Frame-by-frame tracking evaluation
- Cross-subject and cross-sequence comparisons

### Articulator Tracking
- **Temporal tracking**: Maintain consistency across video frames
- **Incisor tracking**: Specialized algorithms for upper and lower incisors
- **Arytenoid muscle tracking**: Laryngeal structure monitoring
- **Gravity algorithm**: Enhanced tongue tip tracking with reference curves
- Optimization-based tracking with boundary constraints

### Post-Processing Methods
- **B-spline regularization**: Smooth contour refinement
- **Graph-based methods**: Active contour evolution
- **Skeletonization**: Centerline extraction
- **Contour refinement**: Adaptive thresholding with retry logic
- **Region filling**: Handle incomplete masks
- **Upscaling**: Multi-resolution processing for better accuracy

### Data Management
- Support for multiple image formats (DICOM, PNG, JPEG, NPY)
- RGB mode using temporal slices (t-1, t, t+1) as channels
- Annotation handling from ImageJ ROI files
- Dataset splitting following research protocols
- Exclusion lists for problematic frames
- Case mapping between different framework outputs

### Analysis & Visualization
- Automated results table generation (LaTeX, Markdown, text)
- Statistical analysis with auto-detection
- Dataset statistics and annotation analysis
- Per-sequence and per-subject breakdowns
- Consolidation of tracking and fine-tuning results
- Export contours in multiple formats (NPY, CSV)

---

## Project Structure

```
vocal-tract-seg/
├── train_maskrcnn.py              # Main training script for Mask R-CNN
├── test_maskrcnn.py                # Model testing and evaluation
├── inference_maskrcnn.py           # Run inference on new data
├── dataset.py                      # Dataset classes and data loading
├── evaluation.py                   # Evaluation metrics (P2CP, Jaccard)
├── loss.py                         # Custom loss functions
├── helpers.py                      # Utility functions
├── settings.py                     # Configuration and constants
├── augmentations.py                # Data augmentation pipeline
│
├── create_results_table.py         # Generate statistical analysis tables
├── create_case_mapping.py          # Map nnUNet cases to metadata
├── consolidate_tracking_results.py # Aggregate tracking results
├── consolidate_fine_tuning_results.py # Consolidate fine-tuning experiments
│
├── track_articulators.py           # Track articulators across frames
├── track_upper_incisor.py          # Upper incisor tracking
├── track_lower_incisor.py          # Lower incisor tracking
├── track_incisors/                 # Incisor tracking utilities
│   ├── __init__.py
│   ├── optimization.py
│   └── visualization.py
│
├── config/                         # Configuration files
│   ├── thesis_config/              # Thesis experiment configurations
│   │   ├── 1612_train.yaml
│   │   ├── 1618_train.yaml
│   │   ├── 1640_train.yaml
│   │   └── ...
│   ├── foundation_model.yaml       # Foundation model configs
│   └── inference_nnunet_test.yaml  # nnUNet inference settings
│
├── results/                        # Experimental results
├── logs/                           # Training logs
├── mlruns/                         # Sacred/MLflow experiment tracking
└── requirements.txt                # Python dependencies
```

### Key Files Description

- **train_maskrcnn.py**: Sacred-based training script with experiment tracking
- **dataset.py**: Custom PyTorch datasets for MRI data with annotation loading
- **evaluation.py**: Comprehensive evaluation metrics including P2CP and Jaccard
- **create_results_table.py**: Automated generation of publication-ready tables
- **track_*.py**: Temporal tracking algorithms for maintaining consistency
- **consolidate_*.py**: Scripts for aggregating results across experiments

---

## Experimental Branches

This repository contains multiple experimental branches exploring different segmentation approaches:

### Main Branch
- **Focus**: Mask R-CNN training on ASD1/ASD2 datasets
- **Status**: Production-ready with full evaluation pipeline
- **Features**: Complete training, testing, tracking, and analysis tools

### exp/yolo-seg
- **Focus**: YOLO-based segmentation experiments
- **Models**: YOLOv8, YOLO11 with segmentation heads
- **Features**: Real-time segmentation, dataset conversion, evaluation
- **Scripts**: YOLO training pipeline, inference, and evaluation

### exp/nnnetv2
- **Focus**: nnUNet framework integration
- **Features**: Medical image segmentation, case mapping, preprocessing
- **Output**: Compatible format with Mask R-CNN for comparison
- **Config**: Foundation model configurations

### exp/SAM2-ZERO_SHORT
- **Focus**: Segment Anything Model (SAM) 2.1 zero-shot experiments
- **Prompts**: Box prompts (23.36% IoU), point prompts (2.81% IoU)
- **Goal**: Evaluate foundation models without training
- **Findings**: Multi-point strategies for improved performance

### exp/mask-rcnn-holdout
- **Focus**: Hold-out validation experiments
- **Purpose**: Subject-level cross-validation
- **Analysis**: Generalization across different speakers

### exp/asd1-asd2-training-configs
- **Focus**: Training configurations for ASD1/ASD2 datasets
- **Features**: Data split documentation, batch size optimization
- **Models**: ResNet50-FPN-v2 architecture experiments

### feature/results-table-generator
- **Focus**: Automated analysis and table generation
- **Features**: Auto-detection, LaTeX/Markdown output
- **Status**: Merged to main

### feature/data-statistics-analysis
- **Focus**: Dataset characterization and statistics
- **Analysis**: Annotation coverage, sequence analysis
- **Output**: Comprehensive dataset reports

### feat/data_split
- **Focus**: Data splitting following research protocols
- **Reference**: Follows Table 5.2 methodology
- **Output**: CSV files with sequence information

### fix/training-eval-mode-bug
- **Fix**: Resolved eval mode target passing issue
- **Impact**: Corrected validation behavior
- **Status**: Merged to main

---

## Advanced Usage

### Results Analysis

The repository includes powerful tools for analyzing and presenting results.

#### Basic Statistics

Generate statistics for a single evaluation:

```bash
python create_results_table.py results/test_results.csv
```

#### Compare Subjects

```bash
python create_results_table.py results.csv --compare subject -s1 1618 -s2 1640
```

#### Compare Sequences

```bash
python create_results_table.py results.csv --compare sequence -s1 S10 -s2 S14
```

#### Generate LaTeX Tables

For academic publications:

```bash
# Basic table
python create_results_table.py results.csv -f latex -o results_table.tex

# Cross-subject comparison
python create_results_table.py results.csv --all-subjects -m p2cp_rms -f latex

# Per-sequence breakdown for a subject
python create_results_table.py results.csv --per-sequence 1640 -f latex
```

#### Available Metrics
- `p2cp_rms`: Point-to-Curve-Point Root Mean Square (mm)
- `p2cp_mean`: Mean point-to-curve distance (mm)
- `p2cp_std`: Standard deviation of distances (mm)
- `jaccard`: Jaccard Index (overlap coefficient)

See [README_create_results_table.md](README_create_results_table.md) for detailed documentation.

### Articulator Tracking

Track articulators across temporal sequences to maintain consistency:

```bash
# Track all articulators
python track_articulators.py with config/track_config.yaml

# Track upper incisor specifically
python track_upper_incisor.py with config/upper_incisor_config.yaml

# Track lower incisor specifically
python track_lower_incisor.py with config/lower_incisor_config.yaml
```

#### Tracking Features
- Temporal smoothing across frames
- Boundary constraint optimization
- Reference curve integration (gravity algorithm)
- Adaptive regularization parameters

### Consolidating Results

After running multiple experiments:

```bash
# Consolidate tracking results
python consolidate_tracking_results.py with config/consolidate_config.yaml

# Consolidate fine-tuning experiments
python consolidate_fine_tuning_results.py with config/fine_tuning_config.yaml
```

### nnUNet Integration

Create case mappings for nnUNet framework:

```bash
python create_case_mapping.py \
    --config config/train_config.yaml \
    --output case_mapping.csv \
    --split both
```

This generates mappings between nnUNet case IDs (e.g., `case_0001`) and original metadata (subject/sequence/frame).

### Custom Post-Processing

Edit the `POST_PROCESSING` dictionary in `inference_maskrcnn.py` to customize:

```python
POST_PROCESSING = {
    'tongue': {
        'min_length': 50,
        'smooth_sigma': 2.0,
        'fill_holes': True
    },
    'soft_palate': {
        'min_length': 30,
        'smooth_sigma': 1.5,
        'apply_graph': True
    }
}
```

### Working with Different Data Formats

The dataset supports multiple formats:

```python
# In your config YAML
dataset_config:
  image_format: 'png'  # or 'dicom', 'jpg', 'npy'
  use_rgb_mode: true   # Use (t-1, t, t+1) as RGB channels
  include_background: false
```

---

## Configuration Files

All experiments use YAML configuration files with Sacred experiment tracking.

### Training Configuration Example

```yaml
# config/thesis_config/1640_train.yaml
datadir: /path/to/data
train_sequences:
  1612: [S1, S2, S3, S4]
  1618: [S1, S2, S3]
valid_sequences:
  1635: [S1]
test_sequences:
  1640: [S1, S2]

classes:
  - tongue
  - upper_lip
  - lower_lip
  - pharynx
  - soft_palate
  - epiglottis
  - vocal_folds
  - arytenoid
  - thyroid

model:
  name: maskrcnn_resnet50_fpn_v2
  pretrained: true

training:
  batch_size: 8
  n_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0005
  scheduler: reduce_on_plateau

augmentation:
  horizontal_flip: true
  vertical_flip: true
  gaussian_blur: true
```

### Inference Configuration Example

```yaml
# config/inference_config.yaml
model_path: /path/to/best_model.pt
data_dir: /path/to/test/data
output_dir: /path/to/output
sequences:
  1640: [S1, S2, S3]

post_processing:
  apply_smoothing: true
  apply_tracking: true
  min_contour_length: 30
```

---

## Evaluation Metrics

### P2CP (Point-to-Curve-Point) Distance

Measures the distance from each point on the predicted contour to the closest point on the ground truth contour.

- **P2CP_RMS**: Root mean square distance (primary metric)
- **P2CP_MEAN**: Average distance
- **P2CP_STD**: Standard deviation of distances

Lower values indicate better performance. Typical values: 1-3mm for good performance.

### Jaccard Index

Measures the overlap between predicted and ground truth masks:

```
Jaccard = |A ∩ B| / |A ∪ B|
```

Range: [0, 1], where 1 indicates perfect overlap.

### Per-Articulator Performance

Different articulators have different difficulty levels:
- **Tongue**: Largest articulator, typically best performance (P2CP ~1-2mm)
- **Lips**: Good performance due to clear boundaries (P2CP ~1-2mm)
- **Pharynx**: Challenging due to low contrast (P2CP ~2-3mm)
- **Small structures** (arytenoid, vocal folds): Most challenging (P2CP ~2-4mm)

---

## Output Files

### Training Outputs
- `best_model.pt`: Best model checkpoint based on validation loss
- `mlruns/`: Sacred experiment logs with metrics and configurations
- `logs/`: Training logs with loss curves
- `tensorboard/`: TensorBoard visualization files (if enabled)

### Testing Outputs
- `test_results.csv`: Per-frame evaluation metrics
  - Columns: subject, sequence, frame, articulator, p2cp_rms, p2cp_mean, p2cp_std, jaccard
- `inference_contours/`: Predicted contour coordinates (.npy format)
- `visualization/`: Optional visualization images

### Tracking Outputs
- `optimization_*.csv`: Tracking optimization results per frame
- `tracked_contours/`: Temporally smoothed contour sequences
- `tracking_metrics.csv`: Consistency metrics across frames

---

## Tips and Best Practices

### Training
1. **Start with pretrained weights**: Use `pretrained: true` for faster convergence
2. **Monitor validation metrics**: Early stopping prevents overfitting
3. **Adjust batch size**: Larger batches (8-16) typically work better
4. **Use data augmentation**: Helps with limited training data
5. **Subject-level splits**: Ensure validation subjects are different from training

### Evaluation
1. **Multiple runs**: Run evaluation multiple times for statistical significance
2. **Per-articulator analysis**: Some articulators are consistently harder
3. **Visual inspection**: Always check some predictions visually
4. **Temporal consistency**: Check frame-to-frame variation

### Tracking
1. **Tune regularization**: Balance temporal smoothness vs. frame accuracy
2. **Use reference curves**: Gravity algorithm helps with tongue tip
3. **Check boundary constraints**: Prevent unrealistic movements
4. **Optimize per articulator**: Different structures need different parameters

### Common Issues
- **Low Jaccard Index**: Check mask post-processing, may need hole filling
- **High P2CP at boundaries**: Increase contour smoothing
- **Temporal jitter**: Increase tracking regularization weight
- **Missing detections**: Lower confidence threshold or add more training data

---

## Requirements

Key dependencies (see `requirements.txt` for complete list):

```
torch>=1.7.0
torchvision>=0.8.0
sacred>=0.8.2
pandas>=1.2.0
numpy>=1.19.0
scipy>=1.6.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
roifile>=2020.11.28
opencv-python>=4.5.0
```

External dependencies:
- `vt_tools`: Vocal tract analysis utilities
- `vt_tracker`: Articulator tracking algorithms

---

## Acknowledgments

This work is based on the original implementation by **Vinicius de Paulo Souza Ribeiro**.

**Original Author:**  
vinicius.ribeiro1@gmail.com  
[LinkedIn](https://www.linkedin.com/in/vribeiro1/)  
[Website](https://vribeiro1.github.io)

### Related Publications

<ul>

<li>
<b>Deep Supervision of the Vocal Tract Shape for Articulatory Synthesis of Speech</b><br>
Vinicius Ribeiro<br>
Ph.D. Thesis
</li>

<li>
<b>Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging</b><br>
Vinicius Ribeiro, Karyna Isaieva, Justine Leclere, Jacques Felblinger, Pierre-André Vuissoz, Yves Laprie<br>
Nov 10, 2023 <a href="https://vribeiro1.github.io/publications#:~:text=Computer%20Methods%20and%20Programs%20in%20Biomedicine">Computer Methods and Programs in Biomedicine</a>
</li>

</ul>

### External dependencies

This repository requires vt_tools (<a href="https://github.com/vribeiro1/vt_tools">Github</a>, <a href="https://gitlab.inria.fr/vsouzari/vt_tools">Gitlab</a>) and vt_tracker (<a href="https://gitlab.inria.fr/vsouzari/vt_tracker">Gitlab</a>). To install the library, follow the instructions bellow.

<ol>

<li>Clone the repos</li>

```
>>> git clone git@gitlab.inria.fr:vsouzari/vt_tools.git
>>> git clone git@gitlab.inria.fr:vsouzari/vt_tracker.git
```

<li>Install the repo</li>

```
>>> pip3 install -e /path/to/vt_tools
>>> pip3 install -e /path/to/vt_tracker
```

</ol>
