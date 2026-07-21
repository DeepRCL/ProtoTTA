#!/usr/bin/env python3
"""Run the seven-stage deterministic/adaptive-lambda validation protocol."""

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PF_SCRIPT = ROOT / 'ProtoPFormer' / 'evaluate_robustness_dogs.py'
PV_SCRIPT = ROOT / 'ProtoViT' / 'evaluate_robustness.py'
PF_MODEL = ROOT / 'ProtoPFormer' / 'output_cosine' / 'Dogs' / 'deit_small_patch16_224' / '1028-adamw-0.05-200-protopformer' / 'checkpoints' / 'epoch-best.pth'
PF_DATA = ROOT / 'ProtoPFormer' / 'datasets' / 'stanford_dogs_c'
PV_MODEL = ROOT / 'ProtoViT' / 'saved_models' / 'deit_small_patch16_224' / 'exp1' / '14finetuned0.8609.pth'
PV_DATA = ROOT / 'ProtoViT' / 'datasets' / 'cub200_c'
OUT = ROOT / 'protocol_results_v2'


def run(arguments):
    command = [sys.executable, '-u', *map(str, arguments)]
    print('\n$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def pf_common(gpu, seed):
    return [
        PF_SCRIPT, '--model', PF_MODEL, '--data_dir', PF_DATA,
        '--severity', '5', '--batch_size', '128', '--num_workers', '4',
        '--proto_threshold', '0.55', '--proto_mapping', 'sigmoid',
        '--proto_sigmoid_center', '1.0', '--proto_sigmoid_temp', '1.0',
        '--proto_no_importance', '--proto_branch', 'both',
        '--seed', str(seed), '--gpuid', gpu, '--overwrite',
    ]


def pv_common(gpu, seed):
    return [
        PV_SCRIPT, '--model', PV_MODEL, '--data_dir', PV_DATA,
        '--batch_size', '128', '--seed', str(seed), '--gpuid', gpu, '--overwrite',
    ]


def accuracy_map(path, mode):
    with path.open() as handle:
        payload = json.load(handle)
    return {
        corruption: result['5']['accuracy']
        for corruption, result in payload['results'][mode].items()
    }


def step1():
    files = [
        PF_SCRIPT, ROOT / 'ProtoPFormer' / 'proto_tta.py',
        ROOT / 'ProtoPFormer' / 'sar_adapt.py', PV_SCRIPT,
        ROOT / 'ProtoViT' / 'proto_entropy.py', ROOT / 'ProtoViT' / 'tent.py',
        ROOT / 'ProtoViT' / 'sar_adapt.py', ROOT / 'analyze_tta_results.py',
    ]
    for path in files:
        py_compile.compile(str(path), doraise=True)
    print('Step 1 passed: all deterministic/adaptive-lambda modules compile.')


def step2(gpu, seed):
    paths = []
    for repeat in ('a', 'b'):
        output = OUT / f'pf_frost_equivalence_{repeat}.json'
        paths.append(output)
        run(pf_common(gpu, seed) + [
            '--output', output, '--corruptions', 'frost',
            '--modes', 'proto_tta', 'proto_tta_plus_7030', '--proto_lambda', '0.7',
        ])
    maps = [
        accuracy_map(path, mode)
        for path in paths for mode in ('proto_tta', 'proto_tta_plus_7030')
    ]
    if not all(item == maps[0] for item in maps[1:]):
        raise RuntimeError(f'Determinism/equivalence failed: {maps}')
    print(f'Step 2 passed exactly: {maps[0]}')


def step3(gpu, seed):
    run(pf_common(gpu, seed) + [
        '--output', OUT / 'pf_unified_baselines.json',
        '--modes', 'normal', 'tent', 'sar', 'eata', 'proto_tta_plus_7030',
        '--lr', '0.001', '--steps', '1', '--sar-lr', '0.0001',
        '--sar-margin', '2.8724950456692273', '--sar-reset', '0.2',
        '--sar-rho', '0.05', '--proto_lambda', '0.7',
    ])


def step4(gpu, seed):
    for value in (0.0, 0.25, 0.5, 0.7, 0.75, 1.0):
        run(pf_common(gpu, seed) + [
            '--output', OUT / f'pf_lambda_{value}.json',
            '--modes', 'proto_tta', '--proto_lambda', str(value),
        ])


def step5(gpu, seed):
    for value in (0.0, 0.25, 0.5, 0.7, 0.75, 1.0):
        run(pv_common(gpu, seed) + [
            '--output', OUT / f'pv_lambda_{value}.json',
            '--modes', 'proto_imp_conf_v3', '--proto-lambda', str(value),
        ])


def step6(gpu, seed):
    run(pf_common(gpu, seed) + [
        '--output', OUT / 'pf_frost_diagnostics.json', '--corruptions', 'frost',
        '--modes', 'proto_tta', '--proto_lambda', '0.7',
        '--proto_record_diagnostics',
    ])
    run(pv_common(gpu, seed) + [
        '--output', OUT / 'pv_frost_diagnostics.json', '--corruptions', 'frost',
        '--modes', 'proto_imp_conf_v3',
        '--proto-lambda', '0.7', '--proto-record-diagnostics',
    ])


def step7(gpu, seed):
    # This is the manuscript rule: current-batch top-target activation margin,
    # delta0=.25, no EMA clipping, and identical filtering/confidence weighting.
    run(pf_common(gpu, seed) + [
        '--output', OUT / 'pf_adaptive_margin.json',
        '--modes', 'normal', 'tent', 'eata', 'proto_tta_adaptive',
        '--proto_shared_confidence_weighting',
        '--proto_adaptive_strategy', 'activation_margin',
        '--proto_adaptive_delta0', '0.25', '--proto_adaptive_topk', '3',
        '--proto_lambda_ema_momentum', '0', '--proto_lambda_min', '0',
        '--proto_lambda_max', '1', '--proto_record_diagnostics',
    ])
    run(pv_common(gpu, seed) + [
        '--output', OUT / 'pv_adaptive_margin.json',
        '--modes', 'normal', 'tent', 'eata', 'proto_imp_conf_adaptive',
        '--proto-shared-confidence-weighting',
        '--proto-adaptive-strategy', 'activation_margin',
        '--proto-adaptive-delta0', '0.25', '--proto-adaptive-topk', '3',
        '--proto-lambda-ema-momentum', '0', '--proto-lambda-min', '0',
        '--proto-lambda-max', '1', '--proto-record-diagnostics',
    ])


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=range(1, 8), required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == 1:
        STEPS[1]()
    else:
        STEPS[args.step](args.gpu, args.seed)


if __name__ == '__main__':
    main()
