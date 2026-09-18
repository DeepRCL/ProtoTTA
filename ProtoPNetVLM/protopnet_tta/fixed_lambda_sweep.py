#!/usr/bin/env python3
"""Run and summarize the fixed-lambda ProtoTTA+ SICAPv2-C sweep."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


LAMBDAS = [index / 10 for index in range(11)]
SEEDS = [0, 1, 2]
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform',
]
REFERENCE_GAUSSIAN_LAMBDA_07 = 0.5834118755890669


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the reproducible ProtoPNet ProtoTTA+ fixed-lambda sweep'
    )
    parser.add_argument(
        '--model',
        default='./saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth',
    )
    parser.add_argument('--data-dir', default='/mnt/ext/SICAPv2_c/')
    parser.add_argument(
        '--clean-data-dir',
        default='./datasets/SICAPv2_cropped/test_cropped',
    )
    parser.add_argument(
        '--output-dir',
        default='./fixed_lambda_sweep_sicapv2c',
    )
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpuid', default='0')
    parser.add_argument('--smoke-tolerance', type=float, default=0.01)
    parser.add_argument('--skip-smoke', action='store_true')
    parser.add_argument('--skip-prototype-metrics', action='store_true')
    parser.add_argument('--skip-efficiency', action='store_true')
    parser.add_argument(
        '--task-id',
        type=int,
        default=None,
        help='Run one array task (0-38): 33 lambda jobs then 6 baselines',
    )
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def lambda_label(value):
    return f'{value:.1f}'


def result_path(output_dir, kind, seed, lambda_proto=None):
    if kind == 'ProtoHybrid':
        name = f'lambda_{lambda_label(lambda_proto)}_seed_{seed}.json'
    else:
        name = f'{kind.lower()}_seed_{seed}.json'
    return output_dir / 'raw' / name


def jobs():
    sweep_jobs = [
        ('ProtoHybrid', seed, lambda_proto)
        for seed in SEEDS
        for lambda_proto in LAMBDAS
    ]
    baseline_jobs = [
        (mode, seed, None)
        for seed in SEEDS
        for mode in ('Tent', 'EATA')
    ]
    return sweep_jobs + baseline_jobs


def evaluator_command(args, mode, seed, output, corruptions, lambda_proto=None):
    command = [
        sys.executable, '-m', 'protopnet_tta.evaluate_robustness',
        '--model', args.model,
        '--data_dir', args.data_dir,
        '--clean_data_dir', args.clean_data_dir,
        '--output', str(output),
        '--severity', str(args.severity),
        '--batch_size', str(args.batch_size),
        '--seed', str(seed),
        '--gpuid', args.gpuid,
        '--modes', mode,
        '--corruptions', *corruptions,
    ]
    if lambda_proto is not None:
        command.extend(['--lambda-proto', lambda_label(lambda_proto)])
    if not args.skip_prototype_metrics:
        command.append('--prototype-metrics')
    if not args.skip_efficiency:
        command.append('--track-efficiency')
    return command


def run_evaluator(args, mode, seed, output, corruptions, lambda_proto=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = evaluator_command(
        args, mode, seed, output, corruptions, lambda_proto
    )
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )


def load_accuracy(path, mode, corruption, severity):
    with path.open() as handle:
        payload = json.load(handle)
    return payload['results'][mode][corruption][str(severity)]['accuracy']


def run_smoke_test(args, output_dir):
    smoke_path = output_dir / 'smoke' / 'lambda_0.7_seed_0_gaussian_noise.json'
    if smoke_path.exists():
        smoke_path.unlink()
    run_evaluator(
        args,
        'ProtoHybrid',
        0,
        smoke_path,
        ['gaussian_noise'],
        0.7,
    )
    accuracy = load_accuracy(
        smoke_path, 'ProtoHybrid', 'gaussian_noise', args.severity
    )
    difference = abs(accuracy - REFERENCE_GAUSSIAN_LAMBDA_07)
    print(
        f'lambda=0.7 smoke accuracy={accuracy:.8f}; '
        f'reference={REFERENCE_GAUSSIAN_LAMBDA_07:.8f}; '
        f'absolute difference={difference:.8f}',
        flush=True,
    )
    if difference > args.smoke_tolerance:
        raise RuntimeError(
            'lambda=0.7 smoke test failed; stopping before interpreting the sweep'
        )


def read_run(path, mode, severity):
    with path.open() as handle:
        payload = json.load(handle)
    mode_results = payload['results'][mode]
    accuracies = {}
    weighted_sum = 0.0
    weighted_count = 0
    for corruption in CORRUPTIONS:
        result = mode_results[corruption][str(severity)]
        accuracies[corruption] = result['accuracy'] * 100.0
        stats = result.get('adaptation_stats', {})
        mean_weight = stats.get('mean_selected_prototype_weight')
        count = stats.get('selected_prototype_weight_count', 0)
        if mean_weight is not None and count:
            weighted_sum += mean_weight * count
            weighted_count += count
    return {
        'per_corruption_percent': accuracies,
        'total_percent': statistics.fmean(accuracies.values()),
        'selected_weight_sum': weighted_sum,
        'selected_weight_count': weighted_count,
    }


def summarize(args, output_dir):
    lambda_runs = {}
    for lambda_proto in LAMBDAS:
        label = lambda_label(lambda_proto)
        lambda_runs[label] = {
            seed: read_run(
                result_path(output_dir, 'ProtoHybrid', seed, lambda_proto),
                'ProtoHybrid',
                args.severity,
            )
            for seed in SEEDS
        }

    baseline_runs = {
        mode: {
            seed: read_run(
                result_path(output_dir, mode, seed),
                mode,
                args.severity,
            )
            for seed in SEEDS
        }
        for mode in ('Tent', 'EATA')
    }

    lambda_summary = {}
    for label, seed_runs in lambda_runs.items():
        raw_totals = {
            str(seed): run['total_percent']
            for seed, run in seed_runs.items()
        }
        per_corruption = {
            corruption: statistics.fmean(
                run['per_corruption_percent'][corruption]
                for run in seed_runs.values()
            )
            for corruption in CORRUPTIONS
        }
        selected_weight_sum = sum(
            run['selected_weight_sum'] for run in seed_runs.values()
        )
        selected_weight_count = sum(
            run['selected_weight_count'] for run in seed_runs.values()
        )
        lambda_summary[label] = {
            'mean_total_percent': statistics.fmean(raw_totals.values()),
            'std_total_percent': statistics.stdev(raw_totals.values()),
            'raw_seed_totals_percent': raw_totals,
            'per_corruption_mean_percent': per_corruption,
            'mean_selected_prototype_weight': (
                selected_weight_sum / selected_weight_count
                if selected_weight_count else None
            ),
            'selected_prototype_weight_count': selected_weight_count,
        }

    baseline_summary = {}
    for mode, seed_runs in baseline_runs.items():
        raw_totals = {
            str(seed): run['total_percent']
            for seed, run in seed_runs.items()
        }
        baseline_summary[mode] = {
            'mean_total_percent': statistics.fmean(raw_totals.values()),
            'std_total_percent': statistics.stdev(raw_totals.values()),
            'raw_seed_totals_percent': raw_totals,
            'per_corruption_mean_percent': {
                corruption: statistics.fmean(
                    run['per_corruption_percent'][corruption]
                    for run in seed_runs.values()
                )
                for corruption in CORRUPTIONS
            },
        }

    best_lambda = max(
        lambda_summary,
        key=lambda label: lambda_summary[label]['mean_total_percent'],
    )
    per_corruption_best = {}
    for corruption in CORRUPTIONS:
        best_for_corruption = max(
            lambda_summary,
            key=lambda label: lambda_summary[label][
                'per_corruption_mean_percent'
            ][corruption],
        )
        per_corruption_best[corruption] = {
            'lambda': float(best_for_corruption),
            'accuracy_percent': lambda_summary[best_for_corruption][
                'per_corruption_mean_percent'
            ][corruption],
        }
    oracle_mean = statistics.fmean(
        entry['accuracy_percent'] for entry in per_corruption_best.values()
    )

    comparison_values = {
        'Tent': baseline_summary['Tent']['mean_total_percent'],
        'EATA': baseline_summary['EATA']['mean_total_percent'],
        'lambda_0.0': lambda_summary['0.0']['mean_total_percent'],
        'lambda_0.7': lambda_summary['0.7']['mean_total_percent'],
        'lambda_1.0': lambda_summary['1.0']['mean_total_percent'],
    }
    best_total = lambda_summary[best_lambda]['mean_total_percent']
    comparisons = {
        name: {
            'mean_total_percent': value,
            'best_fixed_lambda_delta_percent_points': best_total - value,
        }
        for name, value in comparison_values.items()
    }

    summary = {
        'provenance': {
            'model': args.model,
            'data_dir': args.data_dir,
            'clean_data_dir': args.clean_data_dir,
            'severity': args.severity,
            'batch_size': args.batch_size,
            'seeds': SEEDS,
            'lambdas': LAMBDAS,
            'corruptions': CORRUPTIONS,
            'std_definition': 'sample standard deviation across three seeds',
            'selection_note': (
                'Best fixed lambda is a test-set sweep/ablation, not a '
                'label-free adaptive result.'
            ),
        },
        'lambda_results': lambda_summary,
        'baselines': baseline_summary,
        'descriptively_best_fixed_lambda': {
            'lambda': float(best_lambda),
            **lambda_summary[best_lambda],
        },
        'per_corruption_best_lambda': per_corruption_best,
        'corruption_wise_oracle_mean_percent': oracle_mean,
        'comparisons': comparisons,
    }
    summary_path = output_dir / 'summary.json'
    with summary_path.open('w') as handle:
        json.dump(summary, handle, indent=2)
    print(f'Summary saved to {summary_path}', flush=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize(args, output_dir)
        return

    all_jobs = jobs()
    if args.task_id is not None:
        if not 0 <= args.task_id < len(all_jobs):
            raise ValueError(f'--task-id must be between 0 and {len(all_jobs) - 1}')
        mode, seed, lambda_proto = all_jobs[args.task_id]
        run_evaluator(
            args,
            mode,
            seed,
            result_path(output_dir, mode, seed, lambda_proto),
            CORRUPTIONS,
            lambda_proto,
        )
        return

    if not args.skip_smoke:
        run_smoke_test(args, output_dir)
    for mode, seed, lambda_proto in all_jobs:
        run_evaluator(
            args,
            mode,
            seed,
            result_path(output_dir, mode, seed, lambda_proto),
            CORRUPTIONS,
            lambda_proto,
        )
    summarize(args, output_dir)


if __name__ == '__main__':
    main()
