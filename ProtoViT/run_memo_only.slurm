#!/bin/bash
#SBATCH --job-name=protovit_memo_full_metrics
#SBATCH --output=logs/memo_full_metrics_%j.out
#SBATCH --error=logs/memo_full_metrics_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=main

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Print GPU information
nvidia-smi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate protovit

# Change to project directory
cd /home/mahdi.abootorabi/protovit/ProtoViT/

# Create logs directory if it doesn't exist
mkdir -p logs

# Run MEMO evaluation with full prototype metrics
# This will add all metrics (PAC, PCA, sparsity, calibration, etc.) to MEMO results
python evaluate_robustness_memo_only.py \
    --model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    --data_dir ./datasets/cub200_c/ \
    --clean_data_dir ./datasets/cub200_cropped/test_cropped/ \
    --input ./robustness_results_sev5_metrics.json \
    --output ./robustness_results_sev5_with_memo.json \
    --gpuid 0 \
    --prototype-metrics \
    --use-enhanced-metrics \
    --track-efficiency \
    --proto-baseline-samples 1000 \
    --memo-lr 0.00025 \
    --memo-batch-size 16 \
    --memo-steps 1

echo "End time: $(date)"
echo "Job completed successfully!"
