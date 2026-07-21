#!/bin/bash
#SBATCH --job-name=protosvit_lambda
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/protosvit/logs/lambda_sweep/sweep_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/protosvit/logs/lambda_sweep/sweep_%j.err

# ProtoSViT λ sweep — Stanford Cars-C, all corruptions, severity 5
# Results saved per λ in results/lambda_sweep/
# Compare: λ=1.0 (ProtoTTA), λ=0.7 (ProtoTTA+), λ=0.0 (logit entropy)

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/protosvit"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protosvit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep logs/lambda_sweep

echo "=================================================="
echo "ProtoSViT λ sweep — Stanford Cars-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

CKPT="logs/train/neurips/stanford_cars_folder_2026-03-26_20-02-51_dino_cars/checkpoints/epoch_076.ckpt"
CARS_C_DIR="/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars_c"
CLEAN_DIR="/home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars"

for LAMBDA in 0.0 0.3 0.5 0.7 1.0; do
    OUTPUT="results/lambda_sweep/cars_c_lambda${LAMBDA}.json"
    echo ""
    echo "---------- λ=${LAMBDA} ----------"
    echo "Output: ${OUTPUT}"

    python -u evaluate_robustness_cars_c.py \
        --ckpt "${CKPT}" \
        --cars_c_dir "${CARS_C_DIR}" \
        --clean_dir "${CLEAN_DIR}" \
        --output "${OUTPUT}" \
        --methods proto_tta \
        --severity 5 \
        --proto_lambda "${LAMBDA}" \
        --num_workers "${SLURM_CPUS_PER_TASK:-4}"

    echo "λ=${LAMBDA} done at $(date)"
done

echo ""
echo "=================================================="
echo "All λ values complete"
echo "End: $(date)"
echo "=================================================="
