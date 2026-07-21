#!/bin/bash
#SBATCH --job-name=protovit_lambda
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoViT/logs/lambda_sweep/sweep_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoViT/logs/lambda_sweep/sweep_%j.err

# ProtoViT λ sweep — CUB-200-C, all corruptions, severity 5
# Results saved per λ in results/lambda_sweep/
# Compare: λ=1.0 (ProtoTTA), λ=0.7 (ProtoTTA+), λ=0.0 (logit entropy)

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoViT"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep logs/lambda_sweep

echo "=================================================="
echo "ProtoViT λ sweep — CUB-200-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# NOTE: use the SAME checkpoint as robustness_results_sev5_metrics.json (the one
# behind the paper table), not saved_models/.../best_model.pth. best_model.pth is
# a stray "highest raw test accuracy across ALL training phases" checkpoint that
# main.py overwrites even during the pre-push warm/joint/slots stages, so it ends
# up holding an UN-PUSHED checkpoint (prototypes never projected onto real training
# patches) with inflated clean accuracy (0.8749 vs 0.8609 for the final, pushed +
# finetuned model). That mismatch alone accounts for the ~5-10pt accuracy gap
# between these sweep results and the paper table.
MODEL="saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
DATA_DIR="./datasets/cub200_c/"

for LAMBDA in 0.0 0.3 0.5 0.7 1.0; do
    OUTPUT="results/lambda_sweep/cub200c_lambda${LAMBDA}.json"
    echo ""
    echo "---------- λ=${LAMBDA} ----------"
    echo "Output: ${OUTPUT}"

    python -u evaluate_robustness.py \
        --model "${MODEL}" \
        --data_dir "${DATA_DIR}" \
        --output "${OUTPUT}" \
        --gpuid 0 \
        --batch_size 128 \
        --modes proto_imp_conf_v3 \
        --proto-lambda "${LAMBDA}"

    echo "λ=${LAMBDA} done at $(date)"
done

echo ""
echo "=================================================="
echo "All λ values complete"
echo "End: $(date)"
echo "=================================================="
