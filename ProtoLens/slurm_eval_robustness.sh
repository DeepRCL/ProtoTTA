#!/bin/bash
#SBATCH --job-name=protolens_robustness
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robustness_eval_%j.out
#SBATCH --error=logs/robustness_eval_%j.err

# ProtoLens Robustness Evaluation SLURM Script
# Evaluates TTA methods on Amazon-C corruptions

# Create logs directory if not exists
mkdir -p logs

# Set working directory
cd /home/mahdi.abootorabi/protovit/ProtoLens

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate protovit

echo "=============================================="
echo "ProtoLens Robustness Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Check GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# ============================================
# Full evaluation with all methods
# ============================================
echo ""
echo "Running full robustness evaluation..."
echo ""

python evaluate_robustness_amazonc.py \
    --methods baseline prototta tent eata sar \
    --learning_rate 0.000005 \
    --adaptation_mode layernorm_attn_bias \
    --geo_filter \
    --geo_threshold 0.1 \
    --sigmoid_temperature 5.0 \
    --output Datasets/Amazon-C/results/robustness_results_main.json \
    --prototype-metrics \
    --track-efficiency \
    --force

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "End time: $(date)"
echo "=============================================="
