#!/bin/bash
#SBATCH --job-name=protolens_diag_multi
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_multi_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_multi_%j.err

# Diagnostic: does queuing >1 corruption/severity combo in a single process
# invocation trigger the collapse, even though each combo passed in isolation?
# aggressive@sev20 was healthy alone (~88% adapt acc). Here we run it again,
# followed by mixed@sev20, in the SAME process/model-reload loop as the real
# sweep. If aggressive collapses here too, the bug is scale/queue-related.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — 2-combo queue test"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep_diag/diag_multi.json"

python -u evaluate_robustness_amazonc.py \
    --methods prototta \
    --geo_filter \
    --geo_threshold 0.1 \
    --sigmoid_temperature 5.0 \
    --adaptation_mode layernorm_attn_bias \
    --proto_lambda 1.0 \
    --corruption_types aggressive mixed \
    --severities 20 \
    --output "${OUTPUT}" \
    --force

echo ""
echo "Diagnostic done at $(date)"
echo "Inspect: ${OUTPUT}"
