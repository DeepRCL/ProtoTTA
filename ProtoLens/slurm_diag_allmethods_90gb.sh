#!/bin/bash
#SBATCH --job-name=protolens_diag_all90
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_all90_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_all90_%j.err

# Run ALL methods (baseline, tent, eata, prototta, sar) at the paper's
# original batch_size=128 on a 90GB MIG slice (instead of the 45GB slice
# that OOM'd during ProtoTTA's grad-enabled forward pass). This tells us:
# 1) Does prototta recover to healthy accuracy with more memory, batch_size
#    unchanged (preserving exact paper config)?
# 2) Do eata/sar (untested so far) also need more memory than 45GB provides?

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — ALL methods @ batch_size=128 on 90GB MIG slice"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUTPUT="results/lambda_sweep_diag/diag_allmethods_90gb.json"

python -u evaluate_robustness_amazonc.py \
    --methods baseline tent eata prototta sar \
    --geo_filter \
    --geo_threshold 0.5 \
    --sigmoid_temperature 5.0 \
    --adaptation_mode layernorm_attn_bias \
    --proto_lambda 1.0 \
    --corruption_types aggressive \
    --severities 20 \
    --batch_size 128 \
    --output "${OUTPUT}" \
    --force

echo ""
echo "Diagnostic done at $(date)"
echo "Inspect: ${OUTPUT}"
