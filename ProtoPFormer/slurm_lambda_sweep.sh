#!/bin/bash
#SBATCH --job-name=ppformer_lambda
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoPFormer/logs/lambda_sweep/sweep_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoPFormer/logs/lambda_sweep/sweep_%j.err

# ProtoPFormer λ sweep — Stanford Dogs-C, severity 5
# Paper match target: proto_tta_plus_7030 mean 57.66% in
#   robustness_results_dogs_sev5_v4.json  (paper table Total 57.7)
#
# Critical paper settings (verified empirically against
# robustness_results_dogs_sev5_v4.json, closest match found):
#   - batch_size 128
#   - proto_threshold 0.55
#   - importance OFF (--proto_no_importance)
#   - sigmoid_center 1.0 (NOTE: proto_tta.py's setup_proto_tta() has a
#     sigmoid_center=2.0 factory default, but mode_config ALWAYS passes
#     'sigmoid_center': args.proto_sigmoid_center explicitly for every
#     proto_tta* mode, so that factory default is dead code here and
#     the actual value is fully controlled by --proto_sigmoid_center.
#     Confirmed empirically: center=2.0 made results WORSE, not better.)
#
# With these settings we get frost=53.32%/defocus_blur=45.87% vs paper's
# 54.18%/45.22% -- within ~1pt. The evaluator now enables deterministic
# cuDNN behavior, so repeated runs on the same software and GPU should be
# substantially more reproducible than the historical paper run.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoPFormer"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lambda_sweep}"
mkdir -p "${OUTPUT_DIR}" logs/lambda_sweep

echo "=================================================="
echo "ProtoPFormer λ sweep — Stanford Dogs-C"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "Config : batch=128 threshold=0.55 importance=off center=1.0 branch=both"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

MODEL="output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth"
DATA_DIR="datasets/stanford_dogs_c"

for LAMBDA in 0.0 0.3 0.5 0.7 1.0; do
    OUTPUT="${OUTPUT_DIR}/dogs_c_lambda${LAMBDA}.json"
    echo ""
    echo "---------- λ=${LAMBDA} ----------"
    echo "Output: ${OUTPUT}"

    python -u evaluate_robustness_dogs.py \
        --model "${MODEL}" \
        --data_dir "${DATA_DIR}" \
        --output "${OUTPUT}" \
        --modes proto_tta \
        --severity 5 \
        --batch_size 128 \
        --proto_threshold 0.55 \
        --proto_mapping sigmoid \
        --proto_sigmoid_center 1.0 \
        --proto_sigmoid_temp 1.0 \
        --proto_no_importance \
        --proto_branch both \
        --proto_lambda "${LAMBDA}" \
        --gpuid 0

    echo "λ=${LAMBDA} done at $(date)"
done

echo ""
echo "=================================================="
echo "All λ values complete"
echo "End: $(date)"
echo "=================================================="
