# Adding MEMO to Robustness Evaluation

## Summary

MEMO (Test Time Robustness via Adaptation and Augmentation) has been added to the robustness evaluation pipeline. MEMO adapts the model to each test sample individually by minimizing the entropy of the marginal distribution over multiple augmented views.

## Changes Made

### 1. Updated `evaluate_robustness.py`
- Added `import memo_adapt`
- Added `setup_memo()` function with optimized settings
- Modified `evaluate_single_combination()` to handle MEMO (requires batch_size=1)
- Added MEMO to the `modes` dictionary with optimized configuration:
  - `lr`: 0.00025 (as in paper)
  - `batch_size`: 16 augmented views (reduced from 64 for speed)
  - `steps`: 1 adaptation step per sample

### 2. Created `evaluate_robustness_memo_only.py`
A standalone script to add MEMO results to existing evaluation JSON files without re-running other methods.

### 3. Created SLURM Job Scripts
- `run_robustness_eval.slurm`: Full evaluation including MEMO
- `run_memo_only.slurm`: MEMO-only evaluation (faster)

## Speed Optimizations for MEMO

MEMO is inherently slow because:
1. It processes one sample at a time (episodic adaptation)
2. It generates multiple augmented views (32-64 per sample)
3. It runs gradient descent on each sample

**Optimizations applied:**
- Reduced augmented views from 64 → 16 (still effective, ~4x faster)
- Keep steps=1 (minimal but sufficient)
- Use efficient data loading (num_workers=8)

**Expected runtime:**
- ~13 corruptions × ~5800 test images × ~0.5-1 sec/image = **1-2 hours per corruption**
- Total for all corruptions: **13-26 hours**

## Usage

### Option 1: Run Full Evaluation (All Methods + MEMO)

```bash
# Submit SLURM job
sbatch run_robustness_eval.slurm

# Or run directly (interactive)
conda activate protovit
cd ProtoViT/
python evaluate_robustness.py \
    --model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    --data_dir ./datasets/cub200_c/ \
    --output ./robustness_results_sev5_with_memo.json \
    --gpuid 0
```

### Option 2: Add MEMO to Existing Results (Faster)

If you already have results for other methods and only want to add MEMO:

```bash
# Submit SLURM job
sbatch run_memo_only.slurm

# Or run directly
conda activate protovit
cd ProtoViT/
python evaluate_robustness_memo_only.py \
    --model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    --data_dir ./datasets/cub200_c/ \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --gpuid 0
```

### Option 3: Run with Custom MEMO Settings

```bash
python evaluate_robustness_memo_only.py \
    --model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    --data_dir ./datasets/cub200_c/ \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --memo-lr 0.00025 \
    --memo-batch-size 32 \
    --memo-steps 1 \
    --gpuid 0
```

## SLURM Job Management

### Submit Job
```bash
sbatch run_memo_only.slurm
```

### Check Job Status
```bash
squeue -u $USER
```

### Monitor Job Output
```bash
# Check output log
tail -f logs/memo_eval_<JOB_ID>.out

# Check error log
tail -f logs/memo_eval_<JOB_ID>.err
```

### Cancel Job
```bash
scancel <JOB_ID>
```

## Output Format

Results are saved in JSON format with the same structure as other methods:

```json
{
  "timestamp": "2026-01-25T...",
  "metadata": {...},
  "results": {
    "normal": {...},
    "tent": {...},
    "memo": {
      "gaussian_noise": {
        "5": {
          "accuracy": 0.752
        }
      },
      "fog": {
        "5": {
          "accuracy": 0.801
        }
      },
      ...
    },
    ...
  }
}
```

## Tips for Faster Evaluation

1. **Use pre-generated corrupted datasets** (avoid `--on_the_fly`)
2. **Reduce augmented views**: `--memo-batch-size 8` (faster but may hurt performance)
3. **Evaluate specific corruptions**: Modify the `corruption_types` list in the script
4. **Use multiple GPUs**: Submit separate jobs per corruption

## Troubleshooting

### Out of Memory
- Reduce `--memo-batch-size` (try 8 or 12)
- Check GPU with `nvidia-smi`

### Job Timeout
- Increase `#SBATCH --time=48:00:00` in SLURM script
- Or run specific corruptions separately

### Resume from Interruption
Both scripts support automatic resumption - they skip already completed evaluations.

```bash
# Just re-run the same command
python evaluate_robustness_memo_only.py \
    --input ./robustness_results_sev5_with_memo.json \
    --output ./robustness_results_sev5_with_memo.json \
    ...
```

## Expected Results

Based on MEMO paper (NeurIPS 2022), MEMO typically achieves:
- **Better performance than Tent** on most corruptions
- **Comparable to or better than EATA/SAR** on severe corruptions
- **Trade-off**: Much slower due to per-sample adaptation

## References

- MEMO Paper: https://arxiv.org/abs/2110.09506
- Code: https://github.com/zhangmarvin/memo
