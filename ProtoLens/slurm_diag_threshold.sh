#!/bin/bash
#SBATCH --job-name=protolens_diag_thresh
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_thresh_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_thresh_%j.err

# OOM is fixed (90GB slice, batch_size=128 unchanged). Now tune geo_threshold:
# 0.5 gave 0% adaptation on aggressive (too strict). proto_tta.py's docstring
# says typical tuned values are 0.05-0.3. Sweep a few to find one that lets
# real adaptation happen (adapt_rate > 0) without collapsing, on the paper's
# exact batch_size=128.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — geo_threshold sweep @ batch_size=128, 90GB slice"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for THRESH in 0.1 0.2 0.3 0.4; do
    OUTPUT="results/lambda_sweep_diag/diag_thresh${THRESH}.json"
    echo ""
    echo "---------- geo_threshold=${THRESH} ----------"
    python -u evaluate_robustness_amazonc.py \
        --methods prototta \
        --geo_filter \
        --geo_threshold "${THRESH}" \
        --sigmoid_temperature 5.0 \
        --adaptation_mode layernorm_attn_bias \
        --proto_lambda 1.0 \
        --corruption_types aggressive qwerty \
        --severities 20 \
        --batch_size 128 \
        --output "${OUTPUT}" \
        --force
done

echo ""
echo "Diagnostic done at $(date)"
