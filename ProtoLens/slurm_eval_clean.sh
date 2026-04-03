#!/bin/bash
#SBATCH --job-name=eval_clean
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_clean_%j.out
#SBATCH --error=logs/eval_clean_%j.err

# Create logs directory if not exists
mkdir -p logs

# Set working directory
cd /home/mahdi.abootorabi/protovit/ProtoLens

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate protovit

echo "=============================================="
echo "ProtoLens Clean Dataset Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Check GPU
nvidia-smi

# Run the evaluation script
python evaluate_amazon_clean.py

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "End time: $(date)"
echo "=============================================="
