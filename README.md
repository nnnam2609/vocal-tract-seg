# Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging

## Author
**Nhat Nam Nguyen (Nam)**  
LORIA - Laboratoire Lorrain de Recherche en Informatique et ses Applications  
Université de Lorraine

> **Note:** This is a fork of the original work by Vinicius de Paulo Souza Ribeiro.  
> Original repository: [vribeiro1/vocal-tract-seg](https://github.com/vribeiro1/vocal-tract-seg)

---

## exp/medsam Branch Overview

This branch provides **MedSAM2 inference + evaluation** for vocal tract images and
produces an `evaluation_results_detailed.csv` with the **same structure** as the YOLO
evaluation output, so you can compare models later.

Key points:
- Uses YOLO-format test images and labels
- Computes P2CP mean/RMS + Jaccard (same logic as YOLO eval)
- Optional per-image visualization panels

---

## Quick Start

### 1) Activate environment

```bash
source /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/vocal-tract-seg/.venv-medsam2/bin/activate
```

### 2) Install/verify dependencies

```bash
pip install opencv-python-headless==4.10.0.84
pip install scipy==1.13.1 funcy==2.0 numba==0.59.1
pip install scikit-image==0.22.0 --no-deps
pip install numpy==1.26.4
```

Notes:
- `vt_tools` and `vt_tracker` are required for post-processing:
  ```bash
  pip install -e /path/to/vt_tools
  pip install -e /path/to/vt_tracker
  ```
- The script also adds repo paths to `sys.path`, so `-e` install is optional if paths are valid.

### 3) Run inference + evaluation

```bash
python inference_medsam2_with_config.py \
  --config config/Nam_exp_01082026/inference_medsam2_test.yaml
```

Output CSV:
```
vocal-tract-seg/inference_output_medsam2_Nam_exp_01232026_all1/evaluation_results_detailed.csv
```

---

## Configuration

Primary config: `config/Nam_exp_01082026/inference_medsam2_test.yaml`

Key fields:

```yaml
model:
  config_name: configs/sam2.1/sam2.1_hiera_l
  checkpoint_path: ./external/MedSAM2/exp_log/MedSAM2_VTS_large/checkpoints/checkpoint.pt
  device: cuda

paths:
  input_folder: ./data_yolo_ribbon_2_px/images/test
  output_folder: ./inference_output_medsam2_Nam_exp_01232026_all1
  ground_truth_folder: ./data_yolo_ribbon_2_px/labels/test

data:
  process_all: false
  specific_images_file: ./test_image_list_233.txt
  specific_images:

evaluation:
  save_csv: true
  save_visualizations: true
  save_per_image: true

postprocessing:
  use_vt_tracker: true
```

### Use all images instead of 233

```yaml
data:
  process_all: true
  specific_images_file:
  specific_images:
```

---

## Re-enable / Disable Visualizations

Visualizations are controlled by:

```yaml
evaluation:
  save_visualizations: true
  save_per_image: true
```

If you disable them:
```yaml
evaluation:
  save_visualizations: false
  save_per_image: false
```

If matplotlib is missing:
```bash
pip install matplotlib
```

---

## Running on a GPU Job (Grid5000)

Example:

```bash
oarstat -f -j <JOB_ID> | grep assigned_hostnames
OAR_JOB_ID=<JOB_ID> oarsh <HOSTNAME> "
  cd /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/vocal-tract-seg &&
  source .venv-medsam2/bin/activate &&
  python inference_medsam2_with_config.py --config config/Nam_exp_01082026/inference_medsam2_test.yaml
"
```

---

## Output Structure

- `evaluation_results_detailed.csv` (per-class metrics)
- `eval_*.png` (per-image visualization panels, if enabled)

CSV columns:
- `subject`, `sequence`, `frame`, `image_name`
- `class_id`, `class_name`
- `jaccard_index`, `has_prediction`, `has_ground_truth`
- `confidence`, `pred_pixels`, `gt_pixels`
- `p2cp_mean`, `p2cp_rms`

---

## Troubleshooting

**MissingConfigException (Hydra):**
- Use `config_name: configs/sam2.1/sam2.1_hiera_l` (works with current MedSAM2 Hydra setup).

**Checkpoint not found:**
- Paths are resolved relative to the config and repo root. Make sure the checkpoint exists:
  `external/MedSAM2/exp_log/MedSAM2_VTS_large/checkpoints/checkpoint.pt`

**vt_tracker import errors:**
- Ensure `vt_tools` and `vt_tracker` are installed or accessible via `PYTHONPATH`.

**GPU warning about Flash Attention:**
- Safe to ignore on GTX 1080 Ti (non-Ampere).

---

## Other Branches (short)

- **main**: Mask R-CNN training + inference + evaluation
- **exp/yolo-seg**: YOLO11 segmentation with ribbon mask conversion
- **exp/nnunetv2**: nnUNetv2 experiments
