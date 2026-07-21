#!/bin/bash
#SBATCH --job-name=protolens_base
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/baselines_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/baselines_%j.err

# ProtoLens baselines — run once, λ-independent.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep logs/lambda_sweep

echo "=================================================="
echo "ProtoLens baselines — Amazon-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep/amazon_c_baselines.json"

python -u evaluate_robustness_amazonc.py \
    --methods baseline tent eata \
    --adaptation_mode layernorm_attn_bias \
    --output "${OUTPUT}"

echo "Baselines done at $(date)"
