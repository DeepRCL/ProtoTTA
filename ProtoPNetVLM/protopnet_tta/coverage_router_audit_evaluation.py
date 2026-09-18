#!/usr/bin/env python3
"""Run controlled ProtoPNet coverage-router diagnostics in new output paths."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = (0, 1, 2)
CORRUPTIONS = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform',
)
METHODS = {
    'ForcedRelativeNoGradNorm': 'ProtoAuditForcedRelative',
    'ForcedRelativeGradNorm': 'ProtoAuditForcedRelativeGradNorm',
    'FixedLambda0.2GradNorm': 'ProtoAuditFixedLambda0.2GradNorm',
    'ForcedNativeGradNorm': 'ProtoAuditForcedNativeGradNorm',
    'AbsoluteConsistencyCoverageRouter': (
        'ProtoAbsoluteConsistencyCoverageRouter'
    ),
    'RelativeEvidenceOnly': 'ProtoAuditForcedRelativeGradNorm',
    'ActivationMarginOnly': 'ProtoAuditForcedNativeGradNorm',
    'RatioOnlyRouter': 'ProtoRatioOnlyRouter',
    'AbsoluteOnlyRouter': 'ProtoAbsoluteOnlyRouter',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='./saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth',
    )
    parser.add_argument('--data-dir', default='/mnt/ext/SICAPv2_c/')
    parser.add_argument(
        '--output-dir',
        default='../coverage_router_absolute_audit_sicapv2c',
    )
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpuid', default='0')
    parser.add_argument('--method', choices=tuple(METHODS) + ('all',), default='all')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument(
        '--prototype-metrics',
        action='store_true',
        help=(
            'Collect the clean prototype baseline and report PAC, PCA-W, and '
            'prediction-stability metrics. By default the audit remains '
            'strictly source-free and skips clean-data evaluation.'
        ),
    )
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def result_path(output_dir, method, seed):
    return output_dir / 'raw' / f'{method}_seed_{seed}.json'


def run_one(args, output_dir, method, seed):
    output = result_path(output_dir, method, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-m', 'protopnet_tta.evaluate_robustness',
        '--model', args.model,
        '--data_dir', args.data_dir,
        '--output', str(output),
        '--severity', str(args.severity),
        '--batch_size', str(args.batch_size),
        '--seed', str(seed),
        '--gpuid', args.gpuid,
        '--modes', METHODS[method],
        '--corruptions', *CORRUPTIONS,
        '--lambda-proto', '0.2',
        '--adaptive-delta0', '0.25',
        '--adaptive-top-k', '3',
        '--track-efficiency',
    ]
    if args.prototype_metrics:
        command.append('--prototype-metrics')
    else:
        command.append('--skip-clean-evaluation')
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )


def read_seed(output_dir, method, seed, severity):
    path = result_path(output_dir, method, seed)
    with path.open() as handle:
        payload = json.load(handle)
    mode_results = payload['results'][METHODS[method]]
    corruptions = {}
    diagnostic_sums = {
        key: 0.0 for key in (
            'q_proto', 'q_out', 'q_proto_over_q_out', 'native_lambda',
            'relative_lambda', 'selected_lambda', 'reliable_fraction',
            'native_path_fraction', 'accepted_samples',
        )
    }
    diagnostic_counts = {key: 0 for key in diagnostic_sums}
    adapted_samples = 0
    total_samples = 0
    for corruption in CORRUPTIONS:
        result = mode_results[corruption][str(severity)]
        stats = result['adaptation_stats']
        adapted_samples += stats['adapted_samples']
        total_samples += stats['total_samples']
        for key, distribution in stats['routing_diagnostics'].items():
            count = distribution['count']
            if count:
                diagnostic_sums[key] += distribution['mean'] * count
                diagnostic_counts[key] += count
        corruptions[corruption] = {
            'accuracy_percent': 100.0 * result['accuracy'],
            'diagnostics': stats['routing_diagnostics'],
            'prototype_scores': stats['prototype_scores'],
            'gradient_normalization': stats[
                'component_gradient_normalization'
            ],
            'mean_proto_gradient_norm': stats['mean_proto_gradient_norm'],
            'mean_output_gradient_norm': stats['mean_output_gradient_norm'],
        }
    total = statistics.fmean(
        item['accuracy_percent'] for item in corruptions.values()
    )
    aggregate = {
        key: (
            diagnostic_sums[key] / diagnostic_counts[key]
            if diagnostic_counts[key] else None
        )
        for key in diagnostic_sums
    }
    aggregate.update({
        'accepted_sample_total': adapted_samples,
        'accepted_sample_fraction': (
            adapted_samples / total_samples if total_samples else None
        ),
    })
    return {
        'total_percent': total,
        'aggregate_diagnostics': aggregate,
        'corruptions': corruptions,
    }


def summarize(output_dir, severity):
    available = {
        method: {
            seed: read_seed(output_dir, method, seed, severity)
            for seed in SEEDS
        }
        for method in METHODS
        if all(result_path(output_dir, method, seed).exists() for seed in SEEDS)
    }
    summary = {
        'protocol': {
            'seeds': list(SEEDS),
            'corruptions': list(CORRUPTIONS),
            'severity': severity,
            'batch_size': 64,
            'output_sign': (
                'positive: L=lambda*L_proto+(1-lambda)*L_out; the supplied '
                'original method and 56.05% equivalence target define '
                'L_out as positive softmax entropy'
            ),
            'existing_references_percent': {
                'relative_without_gradient_normalization': 56.051378718673725,
                'fixed_lambda_0.2_without_gradient_normalization': (
                    56.23263007805892
                ),
            },
        },
        'methods': {},
    }
    diagnostic_report = {}
    for method, seed_runs in available.items():
        totals = {
            str(seed): run['total_percent'] for seed, run in seed_runs.items()
        }
        summary['methods'][method] = {
            'mean_total_percent': statistics.fmean(totals.values()),
            'seed_std_total_percent': statistics.stdev(totals.values()),
            'raw_seed_totals_percent': totals,
            'gradient_normalization': method != 'ForcedRelativeNoGradNorm',
            'mean_selected_lambda': statistics.fmean(
                run['aggregate_diagnostics']['selected_lambda']
                for run in seed_runs.values()
            ),
            'mean_native_lambda': statistics.fmean(
                run['aggregate_diagnostics']['native_lambda']
                for run in seed_runs.values()
            ),
            'mean_relative_lambda': statistics.fmean(
                run['aggregate_diagnostics']['relative_lambda']
                for run in seed_runs.values()
            ),
            'native_batch_fraction': statistics.fmean(
                run['aggregate_diagnostics']['native_path_fraction']
                for run in seed_runs.values()
            ),
            'mean_reliable_fraction': statistics.fmean(
                run['aggregate_diagnostics']['reliable_fraction']
                for run in seed_runs.values()
            ),
            'mean_q_proto': statistics.fmean(
                run['aggregate_diagnostics']['q_proto']
                for run in seed_runs.values()
            ),
            'mean_q_out': statistics.fmean(
                run['aggregate_diagnostics']['q_out']
                for run in seed_runs.values()
            ),
            'mean_q_proto_over_q_out': statistics.fmean(
                run['aggregate_diagnostics']['q_proto_over_q_out']
                for run in seed_runs.values()
            ),
            'mean_accepted_sample_fraction': statistics.fmean(
                run['aggregate_diagnostics']['accepted_sample_fraction']
                for run in seed_runs.values()
            ),
            'accepted_sample_total_all_seeds': sum(
                run['aggregate_diagnostics']['accepted_sample_total']
                for run in seed_runs.values()
            ),
        }
        diagnostic_report[method] = {
            str(seed): run['corruptions'] for seed, run in seed_runs.items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / 'per_corruption_seed_diagnostics.json').open('w') as handle:
        json.dump(diagnostic_report, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not args.summarize_only:
        methods = METHODS if args.method == 'all' else (args.method,)
        seeds = SEEDS if args.seed is None else (args.seed,)
        for method in methods:
            for seed in seeds:
                run_one(args, output_dir, method, seed)
    summarize(output_dir, args.severity)


if __name__ == '__main__':
    main()
