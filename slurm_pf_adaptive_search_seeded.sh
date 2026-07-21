#!/bin/bash
#SBATCH --job-name=pf_adapt_search
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_search_%A_%a.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_search_%A_%a.err

set -euo pipefail

ROOT=/home/mahdi.abootorabi/protovit
RUN_ID=${PF_SEARCH_RUN_ID:-${SLURM_ARRAY_JOB_ID}}
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

python -u "${SCRIPT}" \
    --model "${MODEL}" \
    --data_dir "${DATA}" \
    --output "${OUT}/adaptive_search_seed${SEED}.json" \
    --modes proto_tta_adaptive_search \
    --seed "${SEED}" \
    --severity 5 \
    --batch_size 128 \
    --num_workers 4 \
    --gpuid 0 \
    --overwrite \
    --adapt_model_mode train \
    --lr 1e-3 \
    --steps 1 \
    --proto_threshold 0.55 \
    --proto_mapping sigmoid \
    --proto_sigmoid_center 1.0 \
    --proto_sigmoid_temp 1.0 \
    --proto_no_importance \
    --proto_branch both \
    --proto_shared_confidence_weighting \
    --proto_adaptive_strategy activation_margin \
    --proto_adaptive_delta0 0.25 \
    --proto_adaptive_topk 3 \
    --proto_lambda_ema_momentum 0 \
    --proto_lambda_min 0 \
    --proto_lambda_max 1 \
    --proto_lambda_search_radius 0.1 \
    --proto_lambda_search_teacher_temp 0.5 \
    --proto_lambda_search_min_improvement 0 \
    --proto_record_diagnostics

touch "${OUT}/adaptive_search_seed${SEED}.done"
echo "Completed adaptive-search seed ${SEED} at $(date)"
