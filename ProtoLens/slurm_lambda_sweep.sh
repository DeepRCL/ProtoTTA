#!/bin/bash
#SBATCH --job-name=protolens_lambda
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/sweep_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/sweep_%j.err

# ProtoLens λ sweep — Amazon-C, all corruptions
# Results saved per λ in results/lambda_sweep/
# Compare: λ=1.0 (ProtoTTA), λ=0.7 (ProtoTTA+), λ=0.0 (logit entropy)
#
# NOTE: uses the 90GB MIG slice (not 45GB) — at batch_size=128 the
# grad-enabled ProtoTTA forward pass OOMs on the 45GB slice and silently
# falls back to all-zero logits (this was the real cause of the ~50%
# "collapse" seen previously, nothing to do with lambda/hyperparameters).
# Only running --methods prototta here since baseline/tent/eata/sar already
# match the paper and don't need to be rerun.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep logs/lambda_sweep

echo "=================================================="
echo "ProtoLens λ sweep — Amazon-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

for LAMBDA in 0.0 0.3 0.5 0.7 1.0; do
    OUTPUT="results/lambda_sweep/amazon_c_lambda${LAMBDA}.json"
    echo ""
    echo "---------- λ=${LAMBDA} ----------"
    echo "Output: ${OUTPUT}"

    python -u evaluate_robustness_amazonc.py \
        --methods prototta \
        --learning_rate 0.000005 \
        --adaptation_mode layernorm_attn_bias \
        --geo_filter \
        --geo_threshold 0.1 \
        --sigmoid_temperature 5.0 \
        --batch_size 128 \
        --prototype-metrics \
        --track-efficiency \
        --proto_lambda "${LAMBDA}" \
        --output "${OUTPUT}" \
        --force

    echo "λ=${LAMBDA} done at $(date)"
done

echo ""
echo "=================================================="
echo "All λ values complete"
echo "End: $(date)"
echo "=================================================="
