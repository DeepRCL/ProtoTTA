#!/bin/bash
#SBATCH --job-name=tta_protocol
#SBATCH --partition=mig
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:nvidia_b200_2g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
# At most three tasks may run concurrently because this node exposes three
# concurrent experiment slots requested by this project. Each task uses one
# 45 GB MIG instance; the node currently exposes eight of this profile.
#SBATCH --array=0-29%3
#SBATCH --output=/home/mahdi.abootorabi/protovit/logs/tta_protocol/%A_%a.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/logs/tta_protocol/%A_%a.err

set -euo pipefail

ROOT=/home/mahdi.abootorabi/protovit
RUN_ID=${PROTOCOL_RUN_ID:-${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/protocol_results_slurm/${RUN_ID}
CONDA_ROOT=/home/mahdi.abootorabi/miniconda3

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate protovit
cd "${ROOT}"
mkdir -p "${OUT}"

PF_SCRIPT=${ROOT}/ProtoPFormer/evaluate_robustness_dogs.py
PF_MODEL=${ROOT}/ProtoPFormer/output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth
PF_DATA=${ROOT}/ProtoPFormer/datasets/stanford_dogs_c
PV_SCRIPT=${ROOT}/ProtoViT/evaluate_robustness.py
PV_MODEL=${ROOT}/ProtoViT/saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth
PV_DATA=${ROOT}/ProtoViT/datasets/cub200_c

PF_COMMON=(
    "${PF_SCRIPT}" --model "${PF_MODEL}" --data_dir "${PF_DATA}"
    --severity 5 --batch_size 128 --num_workers 4 --seed 0 --gpuid 0 --overwrite
    --proto_threshold 0.55 --proto_mapping sigmoid
    --proto_sigmoid_center 1.0 --proto_sigmoid_temp 1.0
    --proto_no_importance --proto_branch both
)

PV_COMMON=(
    "${PV_SCRIPT}" --model "${PV_MODEL}" --data_dir "${PV_DATA}"
    --batch_size 128 --seed 0 --gpuid 0 --overwrite
)

run_pf() {
    python -u "${PF_COMMON[@]}" "$@"
}

run_pv() {
    python -u "${PV_COMMON[@]}" "$@"
}

TASK=${SLURM_ARRAY_TASK_ID}
echo "Task ${TASK} starting on $(hostname) at $(date)"
echo "Protocol run ID: ${RUN_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

if [[ ${TASK} -eq 0 ]]; then
    for REPEAT in a b; do
        run_pf --output "${OUT}/pf_equivalence_${REPEAT}.json" \
            --corruptions frost --modes proto_tta proto_tta_plus_7030 \
            --proto_lambda 0.7
    done
    python - "${OUT}/pf_equivalence_a.json" "${OUT}/pf_equivalence_b.json" <<'PY'
import json
import sys

maps = []
for path in sys.argv[1:]:
    with open(path) as handle:
        results = json.load(handle)['results']
    for mode in ('proto_tta', 'proto_tta_plus_7030'):
        maps.append({name: value['5']['accuracy'] for name, value in results[mode].items()})
if not all(item == maps[0] for item in maps[1:]):
    raise SystemExit(f'Exact equivalence failed: {maps}')
print(f'Exact deterministic equivalence passed: {maps[0]}')
PY

elif [[ ${TASK} -ge 1 && ${TASK} -le 5 ]]; then
    PF_MODES=(normal tent sar eata proto_tta_plus_7030)
    MODE=${PF_MODES[$((TASK - 1))]}
    EXTRA=()
    if [[ ${MODE} == sar ]]; then
        EXTRA=(--sar-lr 0.0001 --sar-margin 2.8724950456692273 --sar-reset 0.2 --sar-rho 0.05)
    elif [[ ${MODE} == proto_tta_plus_7030 ]]; then
        EXTRA=(--proto_lambda 0.7)
    fi
    run_pf --output "${OUT}/pf_baseline_${MODE}.json" --modes "${MODE}" "${EXTRA[@]}"

elif [[ ${TASK} -ge 6 && ${TASK} -le 11 ]]; then
    LAMBDAS=(0.0 0.25 0.5 0.7 0.75 1.0)
    LAMBDA=${LAMBDAS[$((TASK - 6))]}
    run_pf --output "${OUT}/pf_lambda_${LAMBDA}.json" \
        --modes proto_tta --proto_lambda "${LAMBDA}"

elif [[ ${TASK} -ge 12 && ${TASK} -le 17 ]]; then
    LAMBDAS=(0.0 0.25 0.5 0.7 0.75 1.0)
    LAMBDA=${LAMBDAS[$((TASK - 12))]}
    run_pv --output "${OUT}/pv_lambda_${LAMBDA}.json" \
        --modes proto_imp_conf_v3 --proto-lambda "${LAMBDA}"

elif [[ ${TASK} -eq 18 ]]; then
    run_pf --output "${OUT}/pf_frost_diagnostics.json" --corruptions frost \
        --modes proto_tta --proto_lambda 0.7 --proto_record_diagnostics

elif [[ ${TASK} -eq 19 ]]; then
    run_pv --output "${OUT}/pv_frost_diagnostics.json" --corruptions frost \
        --modes proto_imp_conf_v3 --proto-lambda 0.7 --proto-record-diagnostics

elif [[ ${TASK} -eq 20 ]]; then
    run_pf --output "${OUT}/pf_adaptive_margin.json" --modes proto_tta_adaptive \
        --proto_shared_confidence_weighting --proto_adaptive_strategy activation_margin \
        --proto_adaptive_delta0 0.25 --proto_adaptive_topk 3 \
        --proto_lambda_ema_momentum 0 --proto_lambda_min 0 --proto_lambda_max 1 \
        --proto_record_diagnostics

elif [[ ${TASK} -eq 21 ]]; then
    run_pv --output "${OUT}/pv_adaptive_margin.json" --modes proto_imp_conf_adaptive \
        --proto-shared-confidence-weighting --proto-adaptive-strategy activation_margin \
        --proto-adaptive-delta0 0.25 --proto-adaptive-topk 3 \
        --proto-lambda-ema-momentum 0 --proto-lambda-min 0 --proto-lambda-max 1 \
        --proto-record-diagnostics

elif [[ ${TASK} -ge 22 && ${TASK} -le 25 ]]; then
    PV_MODES=(normal tent eata sar)
    MODE=${PV_MODES[$((TASK - 22))]}
    run_pv --output "${OUT}/pv_baseline_${MODE}.json" --modes "${MODE}"

elif [[ ${TASK} -eq 26 ]]; then
    run_pf --output "${OUT}/pf_adaptive_margin_gradnorm.json" --modes proto_tta_adaptive \
        --proto_shared_confidence_weighting --proto_gradient_normalize \
        --proto_adaptive_strategy activation_margin --proto_adaptive_delta0 0.25 \
        --proto_adaptive_topk 3 --proto_lambda_ema_momentum 0 \
        --proto_lambda_min 0 --proto_lambda_max 1 --proto_record_diagnostics

elif [[ ${TASK} -eq 27 ]]; then
    run_pv --output "${OUT}/pv_adaptive_margin_gradnorm.json" --modes proto_imp_conf_adaptive \
        --proto-shared-confidence-weighting --proto-gradient-normalize \
        --proto-adaptive-strategy activation_margin --proto-adaptive-delta0 0.25 \
        --proto-adaptive-topk 3 --proto-lambda-ema-momentum 0 \
        --proto-lambda-min 0 --proto-lambda-max 1 --proto-record-diagnostics

elif [[ ${TASK} -eq 28 ]]; then
    run_pf --output "${OUT}/pf_adaptive_relative_gradnorm.json" --modes proto_tta_adaptive \
        --proto_shared_confidence_weighting --proto_gradient_normalize \
        --proto_adaptive_strategy relative_reliability --proto_record_diagnostics

elif [[ ${TASK} -eq 29 ]]; then
    run_pv --output "${OUT}/pv_adaptive_relative_gradnorm.json" --modes proto_imp_conf_adaptive \
        --proto-shared-confidence-weighting --proto-gradient-normalize \
        --proto-adaptive-strategy relative_reliability --proto-record-diagnostics
else
    echo "Unknown task ${TASK}" >&2
    exit 2
fi

touch "${OUT}/task_${TASK}.done"
echo "Task ${TASK} completed at $(date)"
