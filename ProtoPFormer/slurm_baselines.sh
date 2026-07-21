#!/bin/bash
#SBATCH --job-name=ppformer_table
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoPFormer/logs/lambda_sweep/table_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoPFormer/logs/lambda_sweep/table_%j.err

# Recompute every Stanford Dogs-C row in the paper accuracy table under one
# consistent batch-128 evaluation: Unadapted, Tent, SAR, EATA, ProtoTTA+ 70/30.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoPFormer"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-results/table_reproduction}"
mkdir -p "${OUTPUT_DIR}" logs/lambda_sweep

echo "=================================================="
echo "ProtoPFormer table reproduction — Stanford Dogs-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "Common : severity=5 batch=128 lr=1e-3 steps=1"
echo "SAR    : lr=1e-4 margin=0.6*ln(120) reset=0.2 rho=0.05"
echo "Proto+ : lambda=0.7 threshold=0.55 importance=off center=1.0 branch=both"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is required'; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"

MODEL="output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth"
DATA_DIR="datasets/stanford_dogs_c"
OUTPUT="${OUTPUT_DIR}/dogs_c_table_bs128.json"
REPORT_PREFIX="${OUTPUT_DIR}/dogs_c_table_bs128"

python -u evaluate_robustness_dogs.py \
    --model "${MODEL}" \
    --data_dir "${DATA_DIR}" \
    --output "${OUTPUT}" \
    --modes normal tent sar eata proto_tta_plus_7030 \
    --severity 5 \
    --batch_size 128 \
    --lr 1e-3 \
    --steps 1 \
    --sar-lr 1e-4 \
    --sar-margin 2.8724950456692273 \
    --sar-reset 0.2 \
    --sar-rho 0.05 \
    --proto_threshold 0.55 \
    --proto_mapping sigmoid \
    --proto_sigmoid_center 1.0 \
    --proto_sigmoid_temp 1.0 \
    --proto_no_importance \
    --proto_branch both \
    --proto_lambda 0.7 \
    --track-efficiency \
    --gpuid 0

python generate_dogs_report.py \
    --input "${OUTPUT}" \
    --severity 5 \
    --output-prefix "${REPORT_PREFIX}"

echo "Table reproduction done at $(date)"
echo "JSON    : ${OUTPUT}"
echo "Markdown: ${REPORT_PREFIX}_report.md"
echo "LaTeX   : ${REPORT_PREFIX}_tables.tex"
