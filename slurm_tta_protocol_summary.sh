#!/bin/bash
#SBATCH --job-name=tta_summary
# This job only parses JSON files and needs no GPU.
#SBATCH --partition=cpu
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/summary_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/summary_%j.err

set -euo pipefail
shopt -s nullglob

ROOT=/home/mahdi.abootorabi/protovit
if [[ -z ${PROTOCOL_RUN_ID:-} ]]; then
    echo "PROTOCOL_RUN_ID must be exported when submitting this summary job" >&2
    exit 2
fi
OUT=${ROOT}/protocol_results_slurm/${PROTOCOL_RUN_ID}
CONDA_ROOT=/home/mahdi.abootorabi/miniconda3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
cd "${ROOT}"

REPORT=${OUT}/summary.txt
mkdir -p "${OUT}"

{
    echo "TTA protocol summary generated at $(date)"
    echo
    DONE=("${OUT}"/task_*.done)
    echo "Completed array tasks: ${#DONE[@]} / 30"
    for TASK in $(seq 0 29); do
        [[ -f "${OUT}/task_${TASK}.done" ]] || echo "MISSING OR FAILED TASK: ${TASK}"
    done

    echo
    echo "=== ProtoPFormer deterministic baselines ==="
    PF_BASE=("${OUT}"/pf_baseline_*.json)
    if ((${#PF_BASE[@]})); then
        python analyze_tta_results.py --merge "${PF_BASE[@]}" \
            --compare proto_tta_plus_7030 eata
    fi

    echo
    echo "=== ProtoViT deterministic baselines ==="
    PV_BASE=("${OUT}"/pv_baseline_*.json)
    if ((${#PV_BASE[@]})); then
        python analyze_tta_results.py --merge "${PV_BASE[@]}"
    fi

    echo
    echo "=== ProtoPFormer fixed-lambda sweep ==="
    PF_SWEEP=("${OUT}"/pf_lambda_*.json)
    ((${#PF_SWEEP[@]})) && python analyze_tta_results.py "${PF_SWEEP[@]}"

    echo
    echo "=== ProtoViT fixed-lambda sweep ==="
    PV_SWEEP=("${OUT}"/pv_lambda_*.json)
    ((${#PV_SWEEP[@]})) && python analyze_tta_results.py "${PV_SWEEP[@]}"

    echo
    echo "=== ProtoPFormer adaptive variants versus EATA ==="
    PF_ADAPT=("${OUT}"/pf_adaptive_*.json)
    PF_EATA=("${OUT}"/pf_baseline_eata.json)
    if ((${#PF_ADAPT[@]} && ${#PF_EATA[@]})); then
        for FILE in "${PF_ADAPT[@]}"; do
            python analyze_tta_results.py --merge "${PF_EATA[0]}" "${FILE}" \
                --compare proto_tta_adaptive eata
        done
    fi

    echo
    echo "=== ProtoViT adaptive variants versus EATA ==="
    PV_ADAPT=("${OUT}"/pv_adaptive_*.json)
    PV_EATA=("${OUT}"/pv_baseline_eata.json)
    if ((${#PV_ADAPT[@]} && ${#PV_EATA[@]})); then
        for FILE in "${PV_ADAPT[@]}"; do
            python analyze_tta_results.py --merge "${PV_EATA[0]}" "${FILE}" \
                --compare proto_imp_conf_adaptive eata
        done
    fi

    echo
    echo "=== Loss and gradient diagnostics ==="
    DIAGNOSTICS=("${OUT}"/*diagnostics.json)
    ((${#DIAGNOSTICS[@]})) && python analyze_tta_results.py "${DIAGNOSTICS[@]}"
} | tee "${REPORT}"

echo "Summary saved to ${REPORT}"
