#!/bin/bash
set -euo pipefail
ROOT=/home/mahdi.abootorabi/protovit
cd "${ROOT}"
mkdir -p logs/tta_protocol
ARRAY_JOB=$(sbatch --parsable slurm_pf_paper_sweep.sh)
ARRAY_JOB=${ARRAY_JOB%%;*}
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:${ARRAY_JOB}" \
    --export="ALL,PAPER_SWEEP_RUN_ID=${ARRAY_JOB}" slurm_pf_paper_sweep_summary.sh)
SUMMARY_JOB=${SUMMARY_JOB%%;*}
echo "Paper-compatible array: ${ARRAY_JOB}"
echo "Summary job          : ${SUMMARY_JOB}"
echo "Results              : ${ROOT}/paper_sweep_results/${ARRAY_JOB}"
