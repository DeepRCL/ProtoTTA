#!/bin/bash
#SBATCH --job-name=protolens_diag_bs
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_bs_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_bs_%j.err

# Root cause confirmed: batch_size=128 OOMs during ProtoTTA's grad-enabled
# forward pass on this 45GB MIG slice, silently falling back to all-zero
# logits. Test with a much smaller batch size to see if it fits and recovers
# healthy accuracy on the same aggressive_s20 case.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — reduced batch_size to avoid OOM"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for BS in 16 8; do
    OUTPUT="results/lambda_sweep_diag/diag_bs${BS}.json"
    echo ""
    echo "---------- batch_size=${BS} ----------"
    python -u evaluate_robustness_amazonc.py \
        --methods prototta \
        --geo_filter \
        --geo_threshold 0.5 \
        --sigmoid_temperature 5.0 \
        --adaptation_mode layernorm_attn_bias \
        --proto_lambda 1.0 \
        --corruption_types aggressive \
        --severities 20 \
        --batch_size "${BS}" \
        --output "${OUTPUT}" \
        --force
done

echo ""
echo "Diagnostic done at $(date)"
