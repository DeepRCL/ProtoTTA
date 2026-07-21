#!/bin/bash
#SBATCH --job-name=pf_memo
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-38%3
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_memo_%A_%a.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_memo_%A_%a.err

set -euo pipefail

ROOT=/home/mahdi.abootorabi/protovit
RUN_ID=${PF_MEMO_RUN_ID:-${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/paper_sweep_results/${RUN_ID}
CONDA_ROOT=/home/mahdi.abootorabi/miniconda3

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
cd "${ROOT}"
mkdir -p "${OUT}" logs/tta_protocol

CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise speckle_noise
    gaussian_blur defocus_blur fog frost brightness
    jpeg_compression contrast pixelate elastic_transform
)

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$((TASK / 13))
CORRUPTION=${CORRUPTIONS[$((TASK % 13))]}

SCRIPT=${ROOT}/ProtoPFormer/evaluate_robustness_dogs.py
MODEL=${ROOT}/ProtoPFormer/output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth
DATA=${ROOT}/ProtoPFormer/datasets/stanford_dogs_c

# Match the existing ProtoViT MEMO baseline: episodic per-image reset,
# all-parameter SGD, 16 AugMix views, lr=2.5e-4, and one update step.
python -u "${SCRIPT}" \
    --model "${MODEL}" \
    --data_dir "${DATA}" \
    --output "${OUT}/memo_${CORRUPTION}_seed${SEED}.json" \
    --modes memo \
    --corruptions "${CORRUPTION}" \
    --seed "${SEED}" \
    --severity 5 \
    --batch_size 128 \
    --num_workers 4 \
    --gpuid 0 \
    --overwrite \
    --memo-lr 0.00025 \
    --memo-views 16 \
    --memo-steps 1

touch "${OUT}/memo_${CORRUPTION}_seed${SEED}.done"
echo "Completed MEMO seed=${SEED} corruption=${CORRUPTION} at $(date)"
