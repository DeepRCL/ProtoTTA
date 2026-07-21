#!/bin/bash
set -euo pipefail

ROOT=/home/mahdi.abootorabi/protovit
cd "${ROOT}"
mkdir -p logs/tta_protocol

ARRAY_JOB=$(sbatch --parsable slurm_tta_protocol_array.sh)
ARRAY_JOB=${ARRAY_JOB%%;*}
SUMMARY_JOB=$(sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB}" \
    --export="ALL,PROTOCOL_RUN_ID=${ARRAY_JOB}" \
    slurm_tta_protocol_summary.sh)
SUMMARY_JOB=${SUMMARY_JOB%%;*}

echo "Protocol array job : ${ARRAY_JOB}"
echo "Summary job        : ${SUMMARY_JOB}"
echo "Results directory  : ${ROOT}/protocol_results_slurm/${ARRAY_JOB}"
echo "Summary after run  : ${ROOT}/protocol_results_slurm/${ARRAY_JOB}/summary.txt"
echo "Monitor            : squeue -j ${ARRAY_JOB},${SUMMARY_JOB}"
