#!/bin/bash
#SBATCH --job-name=protovit_base
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoViT/logs/lambda_sweep/baselines_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoViT/logs/lambda_sweep/baselines_%j.err

# ProtoViT baselines — run once, λ-independent.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoViT"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep logs/lambda_sweep

echo "=================================================="
echo "ProtoViT baselines — CUB-200-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# NOTE: must match the checkpoint used for the paper table (robustness_results_sev5_metrics.json),
# not best_model.pth -- see slurm_lambda_sweep.sh for why best_model.pth is wrong (un-pushed, inflated accuracy).
MODEL="saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
DATA_DIR="./datasets/cub200_c/"
OUTPUT="results/lambda_sweep/cub200c_baselines.json"

python -u evaluate_robustness.py \
    --model "${MODEL}" \
    --data_dir "${DATA_DIR}" \
    --output "${OUTPUT}" \
    --gpuid 0 \
    --batch_size 128 \
    --modes normal tent eata

echo "Baselines done at $(date)"
