#!/bin/bash
#SBATCH --job-name=pf_sar_seeded
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_sar_%A_%a.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_sar_%A_%a.err

set -euo pipefail

ROOT=/home/mahdi.abootorabi/protovit
RUN_ID=${PF_SAR_RUN_ID:-${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/paper_sweep_results/${RUN_ID}
CONDA_ROOT=/home/mahdi.abootorabi/miniconda3

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
cd "${ROOT}"
mkdir -p "${OUT}" logs/tta_protocol

SCRIPT=${ROOT}/ProtoPFormer/evaluate_robustness_dogs.py
MODEL=${ROOT}/ProtoPFormer/output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth
DATA=${ROOT}/ProtoPFormer/datasets/stanford_dogs_c
SEED=${SLURM_ARRAY_TASK_ID}

# This matches the paper-compatible sweep: train-mode stochastic depth,
# batch 128, one update, and the Dogs-C SAR configuration used by the
# existing table reproduction (0.6 * ln(120) entropy margin).
python -u "${SCRIPT}" \
    --model "${MODEL}" \
    --data_dir "${DATA}" \
    --output "${OUT}/sar_seed${SEED}.json" \
    --modes sar \
    --seed "${SEED}" \
    --severity 5 \
    --batch_size 128 \
    --num_workers 4 \
    --gpuid 0 \
    --overwrite \
    --adapt_model_mode train \
    --steps 1 \
    --sar-lr 1e-4 \
    --sar-margin 2.8724950456692273 \
    --sar-reset 0.2 \
    --sar-rho 0.05

touch "${OUT}/sar_seed${SEED}.done"
echo "Completed SAR seed ${SEED} at $(date)"
