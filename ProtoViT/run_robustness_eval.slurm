#!/bin/bash
#SBATCH --job-name=protovit_robustness_eval
#SBATCH --output=logs/robustness_eval_%j.out
#SBATCH --error=logs/robustness_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

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

# Run the robustness evaluation
python evaluate_robustness.py \
    --model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    --data_dir ./datasets/cub200_c/ \
    --clean_data_dir ./datasets/cub200_cropped/test_cropped/ \
    --output ./robustness_results_sev5_with_memo.json \
    --batch_size 32 \
    --gpuid 0 \
    --prototype-metrics \
    --proto-baseline-samples 1000 \
    --track-efficiency \
    --use-enhanced-metrics

echo "End time: $(date)"
echo "Job completed successfully!"
