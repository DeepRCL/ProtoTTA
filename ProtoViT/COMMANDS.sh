#!/bin/bash
# =============================================================================
# MEMO EVALUATION - COMMAND REFERENCE
# =============================================================================
# This file contains all the commands you need to run MEMO evaluation.
# Copy and paste the commands you need.
# =============================================================================

# =============================================================================
# STEP 0: SETUP (One-time)
# =============================================================================

# Activate conda environment
conda activate protovit

# Go to ProtoViT directory
cd /home/mahdi.abootorabi/protovit/ProtoViT/

# Create logs directory
mkdir -p logs

# =============================================================================
# STEP 1: QUICK TEST (2 minutes) - RECOMMENDED FIRST
# =============================================================================

# Test MEMO on 100 samples to verify everything works
python test_memo_quick.py --gpuid 0

# =============================================================================
# STEP 2: SUBMIT SLURM JOB (RECOMMENDED)
# =============================================================================

# Option A: Add MEMO to existing results (13-26 hours)
sbatch run_memo_only.slurm

# Option B: Run full evaluation with all methods (30-50 hours)
sbatch run_robustness_eval.slurm

# =============================================================================
# STEP 3: MONITOR JOB
# =============================================================================

# Check job status
squeue -u $USER

# Watch output in real-time
tail -f logs/memo_eval_*.out

# Check for errors
tail -f logs/memo_eval_*.err

# Check GPU usage on compute node (after job starts)
# ssh <node_name>
# nvidia-smi

# =============================================================================
# STEP 4: MANAGE JOBS
# =============================================================================

# Cancel a job
scancel <JOB_ID>

# Check completed jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS

# Check specific job details
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize

# =============================================================================
# STEP 5: ANALYZE RESULTS (After completion)
# =============================================================================

# Compare MEMO with other methods
python compare_memo_results.py --input robustness_results_sev5_with_memo.json

# =============================================================================
# ALTERNATIVE: RUN INTERACTIVELY (NOT RECOMMENDED - TAKES LONG TIME)
# =============================================================================

# Only use this if you cannot use SLURM
# This will take 13-26 hours and requires you to keep terminal open

python evaluate_robustness_memo_only.py \
    --model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    --data_dir ./datasets/cub200_c/ \
    --clean_data_dir ./datasets/cub200_cropped/test_cropped/ \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --gpuid 0

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If job times out, just resubmit (it will resume)
sbatch run_memo_only.slurm

# If out of memory, reduce batch size
python evaluate_robustness_memo_only.py \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --memo-batch-size 8 \
    --gpuid 0

# Test on specific corruption only (modify script first)
# Edit evaluate_robustness_memo_only.py line with corruption_types

# =============================================================================
# CUSTOMIZATION
# =============================================================================

# Faster (fewer augmented views, may hurt accuracy slightly)
python evaluate_robustness_memo_only.py \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --memo-batch-size 8 \
    --gpuid 0

# More accurate (more augmented views, slower)
python evaluate_robustness_memo_only.py \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --memo-batch-size 32 \
    --gpuid 0

# =============================================================================
# FILE LOCATIONS
# =============================================================================

# SLURM Scripts:
#   run_memo_only.slurm          - MEMO only (recommended)
#   run_robustness_eval.slurm    - Full evaluation

# Python Scripts:
#   test_memo_quick.py                    - Quick test (100 samples)
#   evaluate_robustness_memo_only.py      - MEMO-only evaluation
#   evaluate_robustness.py                - Full evaluation
#   compare_memo_results.py               - Results analysis

# Documentation:
#   MEMO_COMPLETE_GUIDE.md       - Complete guide
#   MEMO_INTEGRATION.md          - Technical details

# Output:
#   logs/memo_eval_<JOB_ID>.out           - Job output
#   logs/memo_eval_<JOB_ID>.err           - Job errors
#   robustness_results_sev5_with_memo.json - Results

# =============================================================================
