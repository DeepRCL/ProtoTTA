#!/usr/bin/env bash
#SBATCH --job-name=carsc_full
#SBATCH --partition=full
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:b200_full:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mahdi.abootorabi/protovit/protosvit/slurm-%j.out

set -euo pipefail

# --- Path Configuration ---
# Use the directory where the script is located as a base if possible
SCRIPT_DIR="/home/mahdi.abootorabi/protovit/protosvit"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"
CONDA_ENV="${CONDA_ENV:-protosvit}"

# --- Experiment Parameters ---
CKPT="${CKPT:-${SCRIPT_DIR}/logs/train/neurips/stanford_cars_folder_2026-03-26_20-02-51_dino_cars/checkpoints/epoch_076.ckpt}"
CARS_C_DIR="${CARS_C_DIR:-/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars_c}"
CLEAN_DIR="${CLEAN_DIR:-/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/cars_c_v6.json}"
SEVERITY="${SEVERITY:-5}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/robustness}"
mkdir -p "${LOG_DIR}"

RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_FILE="${LOG_DIR}/cars_c_robustness_${RUN_STAMP}_${JOB_TAG}.log"

# Fair-comparison default: keep proto LR identical to baseline LR unless user overrides.
EFFECTIVE_LR="${LR:-2e-4}"
EFFECTIVE_PROTO_LR="${PROTO_LR:-${EFFECTIVE_LR}}"

# --- Conda Setup ---
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "ERROR: Conda not found at ${CONDA_ROOT}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

# Redirect stdout and stderr to a log file AND the Slurm output file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================= START ==========================="
echo "Time:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-None}"
# Adding a check to see if the GPU is actually detected
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU detected"
echo "Checkpoint: ${CKPT}"
echo "Cars-C:     ${CARS_C_DIR}"
echo "Clean:      ${CLEAN_DIR}"
echo "Output:     ${OUTPUT}"
echo "Severity:   ${SEVERITY}"
echo "Modes:      normal tent eata sar proto_tta proto_tta_plus"
echo "Base adapt: ${BASELINE_ADAPT_MODE:-vit_ln_only}"
echo "Proto adapt:${PROTO_ADAPT_MODE:-vit_ln_only}"
echo "LR/protoLR:${EFFECTIVE_LR}/${EFFECTIVE_PROTO_LR}"
echo "Steps:      ${STEPS:-2}"
echo "Proto thr : ${PROTO_THRESHOLD:-0.3}"
echo "Proto conf: ${PROTO_CONF_THRESHOLD:-0.0}"
echo "Proto agr : ${PROTO_AGREEMENT_THRESHOLD:-0.0}"
echo "Proto src : ${PROTO_TARGET_SOURCE:-hybrid}"
echo "Proto c-w : ${PROTO_CONSISTENCY_WEIGHT:-0.05}"
echo "Proto wts : ${PROTO_WEIGHT:-0.5}/${LOGIT_WEIGHT:-0.5}"
echo "Proto rel : ${PROTO_TARGET_REL_THRESHOLD:-0.1}"
echo "============================================================"

# Using the -u flag in Python is good for unbuffered logs
python -u evaluate_robustness_cars_c.py \
  --ckpt "${CKPT}" \
  --cars_c_dir "${CARS_C_DIR}" \
  --clean_dir "${CLEAN_DIR}" \
  --output "${OUTPUT}" \
  --methods normal tent eata sar proto_tta proto_tta_plus \
  --severity "${SEVERITY}" \
  --batch_size "${BATCH_SIZE:-50}" \
  --num_workers "${SLURM_CPUS_PER_TASK:-2}" \
  --lr "${EFFECTIVE_LR}" \
  --proto_lr "${EFFECTIVE_PROTO_LR}" \
  --proto_threshold "${PROTO_THRESHOLD:-0.3}" \
  --proto_weight "${PROTO_WEIGHT:-0.5}" \
  --logit_weight "${LOGIT_WEIGHT:-0.5}" \
  --proto_conf_threshold "${PROTO_CONF_THRESHOLD:-0.0}" \
  --proto_agreement_threshold "${PROTO_AGREEMENT_THRESHOLD:-0.0}" \
  --proto_target_source "${PROTO_TARGET_SOURCE:-hybrid}" \
  --proto_consistency_weight "${PROTO_CONSISTENCY_WEIGHT:-0.05}" \
  --proto_target_rel_threshold "${PROTO_TARGET_REL_THRESHOLD:-0.1}" \
  --baseline_adapt_mode "${BASELINE_ADAPT_MODE:-vit_ln_only}" \
  --proto_adapt_mode "${PROTO_ADAPT_MODE:-vit_ln_only}" \
  --steps "${STEPS:-2}"

echo "========================== END ============================"
echo "Finished at: $(date)"
