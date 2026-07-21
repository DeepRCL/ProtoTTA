#!/bin/bash
#SBATCH --job-name=protosvit_base
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/protosvit/logs/lambda_sweep/baselines_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/protosvit/logs/lambda_sweep/baselines_%j.err

# ProtoSViT baselines — run once, λ-independent.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/protosvit"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protosvit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep logs/lambda_sweep

echo "=================================================="
echo "ProtoSViT baselines — Stanford Cars-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CKPT="logs/train/neurips/stanford_cars_folder_2026-03-26_20-02-51_dino_cars/checkpoints/epoch_076.ckpt"
CARS_C_DIR="/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars_c"
CLEAN_DIR="/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars"
OUTPUT="results/lambda_sweep/cars_c_baselines.json"

python -u evaluate_robustness_cars_c.py \
    --ckpt "${CKPT}" \
    --cars_c_dir "${CARS_C_DIR}" \
    --clean_dir "${CLEAN_DIR}" \
    --output "${OUTPUT}" \
    --methods normal tent eata \
    --severity 5 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}"

echo "Baselines done at $(date)"
