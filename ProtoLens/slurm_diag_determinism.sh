#!/bin/bash
#SBATCH --job-name=protolens_diag_determ
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_determ_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_determ_%j.err

# Diagnostic: with torch.use_deterministic_algorithms(True) + cudnn
# determinism + TF32 disabled now forced in evaluate_robustness_amazonc.py,
# does the solo aggressive_s20 / prototta / lambda=1.0 run recover the
# healthy ~88% result, or does it still collapse to ~50%?

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — full determinism + no-TF32 check"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep_diag/diag_determinism_solo.json"

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u evaluate_robustness_amazonc.py \
    --methods prototta \
    --geo_filter \
    --geo_threshold 0.1 \
    --sigmoid_temperature 5.0 \
    --adaptation_mode layernorm_attn_bias \
    --proto_lambda 1.0 \
    --corruption_types aggressive \
    --severities 20 \
    --output "${OUTPUT}" \
    --force

echo ""
echo "Diagnostic done at $(date)"
echo "Inspect: ${OUTPUT}"
