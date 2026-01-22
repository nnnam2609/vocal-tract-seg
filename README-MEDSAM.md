# MedSAM2 experiment (exp/medsam)

This branch adds a lightweight MedSAM2 setup on top of the main codebase.

What was done
- Cloned MedSAM2 into `external/MedSAM2` for local experimentation.
- Added a YOLO-seg to MedSAM2 NPZ converter: `scripts/medsam/prepare_medsam_npz.py`.
- Ignored MedSAM2 clone and generated NPZ data in `.gitignore`.

Environment setup (from repo root)
```bash
conda create -n medsam2 python=3.12 -y
conda activate medsam2

# PyTorch (match your CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

cd external/MedSAM2
pip install -e ".[dev]"
bash download.sh
```

Prepare NPZ dataset (from repo root)
```bash
python scripts/medsam/prepare_medsam_npz.py \
  --yolo-root data_yolo \
  --output-root data_medsam2/npz \
  --splits train,val,test
```
This creates `data_medsam2/npz/{train,val,test}/*.npz` with `imgs` and `gts` arrays as expected by MedSAM2's `NPZRawDataset`.

Prepare NPZ from original dataset (config-driven)
```bash
python scripts/medsam/prepare_medsam_npz.py \
  --source original \
  --config config/Nam_exp_01082026/foundation_model.yaml \
  --output-root data_medsam2/npz \
  --splits train,val,test
```

Training (from `external/MedSAM2`)
1) Copy an existing config and set the NPZ folder path:
```bash
cp sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml \
   sam2/configs/sam2.1_hiera_tiny512_VTS.yaml
```
Edit `sam2/configs/sam2.1_hiera_tiny512_VTS.yaml`:
```
data:
  train:
    datasets:
    - dataset:
        datasets:
        - video_dataset:
            folder: /ABSOLUTE/PATH/TO/vocal-tract-seg/data_medsam2/npz/train
```

2) Run training:
```bash
python training/train.py -c sam2.1_hiera_tiny512_VTS.yaml --use-cluster 0 --num-gpus 1
```

Notes
- The NPZ converter groups frames by filename prefix (e.g., `ArtSpeech_*_S10_####` becomes one sequence).
- Use `--group-by single` in the converter if you prefer one-image "videos".
