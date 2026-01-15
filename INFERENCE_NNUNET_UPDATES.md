# nnU-Net Inference Updates - Mask R-CNN Format Output

## Summary
Modified `inference_nnunet.py` to automatically generate results in Mask R-CNN compatible format after inference, enabling direct comparison between the two models.

## Changes Made

### 1. Modified `inference_nnunet.py`
**Location:** Lines 495-530 (approximately)

**What Changed:**
- Added automatic generation of Mask R-CNN compatible CSV output
- New output file: `test_results_nnunet.csv`
- Format matches `results/32/test_results.csv` from Mask R-CNN

**New Output Format:**
```
subject, sequence, frame, pred_class, p2cp_rms, p2cp_mean, jaccard_index
```

**Key Features:**
- Only includes valid predictions (has_prediction=True and has_ground_truth=True)
- Sorted by subject, sequence, frame, pred_class for consistency
- Controlled by config parameter: `evaluation.save_maskrcnn_format` (default: True)

### 2. Updated Config File
**File:** `config/Nam_exp_01082026/inference_nnunet_test.yaml`

**Changed:**
```yaml
# OLD:
case_mapping_file: /srv/storage/.../case_mapping_Dataset003.csv

# NEW:
case_mapping_file: /srv/storage/.../case_mapping_Dataset003_test.csv
```

**Why:** The test set needs the test-specific mapping file to correctly map case IDs to subject/sequence/frame information.

## How to Use

### Run Inference with Automatic Mapping
```bash
python inference_nnunet.py --config config/Nam_exp_01082026/inference_nnunet_test.yaml
```

### Expected Output Files
After running inference, you'll find in the output folder:

1. **evaluation_results.csv** - Full detailed results with all columns
   - Includes: image, subject, sequence, frame, class, label, metrics, pixel counts, etc.

2. **test_results_nnunet.csv** - Mask R-CNN compatible format
   - Only includes: subject, sequence, frame, pred_class, p2cp_rms, p2cp_mean, jaccard_index
   - Only valid predictions (where both prediction and ground truth exist)
   - Ready for direct comparison with `results/32/test_results.csv`

3. **visualizations/** - PNG files showing GT vs Prediction overlays (if enabled)

4. **contours/** - NPY files with extracted contours (if enabled)

## Comparison Workflow

### 1. Run nnU-Net Inference
```bash
python inference_nnunet.py --config config/Nam_exp_01082026/inference_nnunet_test.yaml
```

### 2. Compare Results
```python
import pandas as pd

# Load both results
maskrcnn_results = pd.read_csv('vocal-tract-seg/results/32/test_results.csv')
nnunet_results = pd.read_csv('nnunet_data/inference_output_Nam_exp_01082026_test/test_results_nnunet.csv')

# Compare metrics
print("Mask R-CNN Performance:")
print(maskrcnn_results[['p2cp_rms', 'p2cp_mean', 'jaccard_index']].describe())

print("\nnnU-Net Performance:")
print(nnunet_results[['p2cp_rms', 'p2cp_mean', 'jaccard_index']].describe())

# Per-class comparison
comparison = pd.merge(
    maskrcnn_results.groupby('pred_class')[['p2cp_rms', 'p2cp_mean']].mean(),
    nnunet_results.groupby('pred_class')[['p2cp_rms', 'p2cp_mean']].mean(),
    on='pred_class',
    suffixes=('_maskrcnn', '_nnunet')
)
print("\nPer-Class Comparison:")
print(comparison)
```

## Configuration Options

### Disable Mask R-CNN Format Output
Add to your config file:
```yaml
evaluation:
  save_maskrcnn_format: false  # Set to false to disable
```

### Change Output Filename
The output filename is hardcoded to `test_results_nnunet.csv`. To customize, modify line ~522 in `inference_nnunet.py`:
```python
maskrcnn_csv_path = os.path.join(cfg['paths']['output_folder'], 'your_custom_name.csv')
```

## Data Requirements

### Case Mapping File
**Required columns:**
- `case_name` (e.g., "case_00897")
- `subject` (e.g., "1775")
- `sequence` (e.g., "S9")
- `frame` (e.g., "1986")

**Example:**
```csv
case_name,case_id,subject,sequence,frame,original_path
case_00897,0897,1775,S9,1986,/path/to/original/1986.nii.gz
```

### Test Set
- Input images: `nnunet_data/raw/Dataset003_VocalTract/imagesTs/`
- Ground truth: `nnunet_data/raw/Dataset003_VocalTract/labelsTs/`
- Case mapping: `nnunet_data/raw/Dataset003_VocalTract/case_mapping_Dataset003_test.csv`

## Troubleshooting

### "No valid predictions with ground truth found"
**Cause:** All predictions failed or ground truth is missing
**Solution:** Check that `labelsTs/` folder contains corresponding `.nii.gz` files

### "case_name not found in mapping"
**Cause:** Case mapping file doesn't match test images
**Solution:** Regenerate case mapping using `prepare_nnunet_dataset.py`

### Missing subject/sequence/frame in output
**Cause:** Case mapping file not found or path incorrect
**Solution:** Verify `case_mapping_file` path in config points to test set mapping

## Next Steps

1. **Run Full Inference:**
   ```bash
   python inference_nnunet.py --config config/Nam_exp_01082026/inference_nnunet_test.yaml
   ```

2. **Verify Output:**
   - Check that `test_results_nnunet.csv` is created
   - Verify it has the same format as Mask R-CNN results
   - Check row count matches expected test set size

3. **Compare Models:**
   - Load both CSV files
   - Compare overall metrics (mean, std)
   - Compare per-class performance
   - Analyze per-subject/sequence performance

4. **Statistical Analysis:**
   - Paired t-tests per articulator
   - Correlation analysis
   - Error distribution analysis
