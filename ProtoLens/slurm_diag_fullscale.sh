#!/bin/bash
#SBATCH --job-name=protolens_diag_full
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_full_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_full_%j.err

# Diagnostic: reproduce the ORIGINAL broken run exactly (all 5 corruptions x
# 4 severities = 20 combos, lambda=1.0 only) with a completely fresh output
# file and current code, to check whether the June 10 collapse still happens
# today or was a one-off (e.g. transient GPU/MIG contention, stale checkpoint,
# leftover state from a prior run). If this comes back healthy (~81% like the
# paper), the bug is not in the code and we just need to rerun the full sweep.
# If it collapses again, the bug is real and scale-dependent -> bisect further.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — full-scale (20 combo) reproduction, lambda=1.0"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep_diag/diag_fullscale_lambda1.0.json"

python -u evaluate_robustness_amazonc.py \
    --methods prototta \
    --geo_filter \
    --geo_threshold 0.1 \
    --sigmoid_temperature 5.0 \
    --adaptation_mode layernorm_attn_bias \
    --proto_lambda 1.0 \
    --output "${OUTPUT}" \
    --force

echo ""
echo "Diagnostic done at $(date)"
echo "Inspect: ${OUTPUT}"
