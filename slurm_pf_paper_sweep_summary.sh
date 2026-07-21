#!/bin/bash
#SBATCH --job-name=pf_paper_summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_paper_summary_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/pf_paper_summary_%j.err

set -euo pipefail
ROOT=/home/mahdi.abootorabi/protovit
if [[ -z ${PAPER_SWEEP_RUN_ID:-} ]]; then
    echo "PAPER_SWEEP_RUN_ID is required" >&2
    exit 2
fi
OUT=${ROOT}/paper_sweep_results/${PAPER_SWEEP_RUN_ID}
CONDA_ROOT=/home/mahdi.abootorabi/miniconda3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
cd "${ROOT}"
python analyze_pf_paper_sweep.py "${OUT}" | tee "${OUT}/summary.txt"
