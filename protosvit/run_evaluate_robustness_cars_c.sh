#!/usr/bin/env bash
set -euo pipefail

# Example interactive allocation:
# srun --partition=mig --nodelist=rcl-nv2.ece.ubc.ca \
#   --gres=gpu:b200_2g.45gb:1 --cpus-per-task=4 --mem=40G \
#   --time=24:00:00 --pty bash

REPO_ROOT="/home/mahdi.abootorabi/protovit"
SCRIPT_DIR="${REPO_ROOT}/protosvit"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"
CONDA_ENV="${CONDA_ENV:-protosvit}"

CKPT="${CKPT:-${SCRIPT_DIR}/logs/train/neurips/stanford_cars_folder_2026-03-26_20-02-51_dino_cars/checkpoints/epoch_076.ckpt}"
CARS_C_DIR="${CARS_C_DIR:-/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars_c}"
CLEAN_DIR="${CLEAN_DIR:-/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/cars_c_robustness_full.json}"
SEVERITY="${SEVERITY:-5}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/robustness}"
mkdir -p "${LOG_DIR}"

RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/cars_c_robustness_${RUN_STAMP}_${JOB_TAG}.log}"

METHODS=(
  unadapted
  tent
  eata
  sar
  prototta
  prototta_plus_70_30
  prototta_plus_80_20
  prototta_plus_90_10
)

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "Could not find conda.sh under ${CONDA_ROOT}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "ProtoS-ViT Cars-C robustness run"
echo "Started     : $(date)"
echo "Host        : $(hostname)"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Conda env   : ${CONDA_DEFAULT_ENV:-unknown}"
echo "Log file    : ${LOG_FILE}"
echo "Checkpoint  : ${CKPT}"
echo "Cars-C dir  : ${CARS_C_DIR}"
echo "Clean dir   : ${CLEAN_DIR}"
echo "Output JSON : ${OUTPUT}"
echo "Severity    : ${SEVERITY}"
echo "Batch size  : ${BATCH_SIZE:-50}"
echo "Proto LR    : ${PROTO_LR:-1e-4}"
echo "Threshold   : ${PROTO_THRESHOLD:-0.9}"
echo "Adapt mode  : ${ADAPT_MODE:-vit}"
echo "============================================================"

python -u evaluate_robustness_cars_c.py \
  --ckpt "${CKPT}" \
  --cars_c_dir "${CARS_C_DIR}" \
  --clean_dir "${CLEAN_DIR}" \
  --output "${OUTPUT}" \
  --methods "${METHODS[@]}" \
  --severity "${SEVERITY}" \
  --batch_size "${BATCH_SIZE:-50}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" \
  --proto_lr "${PROTO_LR:-1e-4}" \
  --proto_threshold "${PROTO_THRESHOLD:-0.9}" \
  --adapt_mode "${ADAPT_MODE:-vit}" \
  --steps "${STEPS:-1}"

echo "Finished    : $(date)"
