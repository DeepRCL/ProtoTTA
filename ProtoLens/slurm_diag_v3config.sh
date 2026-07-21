#!/bin/bash
#SBATCH --job-name=protolens_diag_v3
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_v3_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_v3_%j.err

# Test the documented "V3 Configuration (Best from experiments)" from
# proto_tta.py's own docstring: consensus_strategy=top_k_mean (now fixed
# in code, was hardcoded to 'max'), threshold=0.5 (was 0.1 in sweep scripts).

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — V3 documented config (top_k_mean, threshold=0.5)"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep_diag/diag_v3config.json"

python -u evaluate_robustness_amazonc.py \
    --methods prototta \
    --geo_filter \
    --geo_threshold 0.5 \
    --sigmoid_temperature 5.0 \
    --adaptation_mode layernorm_attn_bias \
    --proto_lambda 1.0 \
    --corruption_types aggressive mixed qwerty \
    --severities 20 \
    --output "${OUTPUT}" \
    --force

echo ""
echo "Diagnostic done at $(date)"
echo "Inspect: ${OUTPUT}"
