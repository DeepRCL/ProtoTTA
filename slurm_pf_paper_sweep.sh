#!/bin/bash
#SBATCH --job-name=pf_paper_sweep
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-41%3
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_paper_%A_%a.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_paper_%A_%a.err

set -euo pipefail
ROOT=/home/mahdi.abootorabi/protovit
RUN_ID=${PAPER_SWEEP_RUN_ID:-${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/paper_sweep_results/${RUN_ID}
CONDA_ROOT=/home/mahdi.abootorabi/miniconda3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
cd "${ROOT}"
mkdir -p "${OUT}" logs/tta_protocol

SCRIPT=${ROOT}/ProtoPFormer/evaluate_robustness_dogs.py
MODEL=${ROOT}/ProtoPFormer/output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth
DATA=${ROOT}/ProtoPFormer/datasets/stanford_dogs_c
COMMON=(
    "${SCRIPT}" --model "${MODEL}" --data_dir "${DATA}"
    --severity 5 --batch_size 128 --num_workers 4 --gpuid 0 --overwrite
    --adapt_model_mode train
    --proto_threshold 0.55 --proto_mapping sigmoid
    --proto_sigmoid_center 1.0 --proto_sigmoid_temp 1.0
    --proto_no_importance --proto_branch both
)

TASK=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK} -lt 33 ]]; then
    LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
    LAMBDA_INDEX=$((TASK / 3))
    SEED=$((TASK % 3))
    LAMBDA=${LAMBDAS[${LAMBDA_INDEX}]}
    python -u "${COMMON[@]}" --seed "${SEED}" \
        --output "${OUT}/lambda_${LAMBDA}_seed${SEED}.json" \
        --modes proto_tta --proto_lambda "${LAMBDA}"
elif [[ ${TASK} -lt 36 ]]; then
    SEED=$((TASK - 33))
    python -u "${COMMON[@]}" --seed "${SEED}" \
        --output "${OUT}/tent_seed${SEED}.json" --modes tent
elif [[ ${TASK} -lt 39 ]]; then
    SEED=$((TASK - 36))
    python -u "${COMMON[@]}" --seed "${SEED}" \
        --output "${OUT}/eata_seed${SEED}.json" --modes eata
else
    SEED=$((TASK - 39))
    python -u "${COMMON[@]}" --seed "${SEED}" \
        --output "${OUT}/adaptive_seed${SEED}.json" \
        --modes proto_tta_adaptive --proto_shared_confidence_weighting \
        --proto_adaptive_strategy activation_margin --proto_adaptive_delta0 0.25 \
        --proto_adaptive_topk 3 --proto_lambda_ema_momentum 0 \
        --proto_lambda_min 0 --proto_lambda_max 1 --proto_record_diagnostics
fi

touch "${OUT}/task_${TASK}.done"
echo "Completed task ${TASK} at $(date)"
