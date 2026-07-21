#!/bin/bash
#SBATCH --job-name=protolens_diag_base
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_base_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_base_%j.err

# Sanity check: is UNADAPTED baseline accuracy on aggressive_s20 actually
# healthy (~87%) right now, or is something more fundamental (data loading,
# label alignment) broken for this corruption specifically? Also runs
# tent (simple logit-entropy TTA, no geo-filter) for comparison.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — baseline + tent sanity check on aggressive_s20"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep_diag/diag_baseline_check.json"

python -u evaluate_robustness_amazonc.py \
    --methods baseline tent \
    --adaptation_mode layernorm_attn_bias \
    --corruption_types aggressive qwerty \
    --severities 20 \
    --output "${OUTPUT}" \
    --force

echo ""
echo "Diagnostic done at $(date)"
echo "Inspect: ${OUTPUT}"
