#!/bin/bash
#SBATCH --job-name=protolens_diag_repeat
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_repeat_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/lambda_sweep/diag_repeat_%j.err

# Diagnostic: re-run the EXACT solo command that previously gave a healthy
# result (aggressive_s20, prototta alone, lambda=1.0), 3 times back to back,
# each to a fresh output file. If results differ run-to-run despite identical
# code/data/global-seed, the collapse is CUDA non-determinism (bistable TTA
# feedback loop), not a "multi-combo queue" bug.

set -euo pipefail

SCRIPT_DIR="/home/mahdi.abootorabi/protovit/ProtoLens"
CONDA_ROOT="${CONDA_ROOT:-/home/mahdi.abootorabi/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit

cd "${SCRIPT_DIR}"
mkdir -p results/lambda_sweep_diag logs/lambda_sweep

echo "=================================================="
echo "ProtoLens DIAGNOSTIC — repeat solo aggressive_s20 x3 (determinism check)"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for i in 1 2 3; do
    OUTPUT="results/lambda_sweep_diag/diag_repeat_solo_run${i}.json"
    echo ""
    echo "---------- run ${i} ----------"
    python -u evaluate_robustness_amazonc.py \
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
    echo "run ${i} done at $(date)"
done

echo ""
echo "Diagnostic done at $(date)"
