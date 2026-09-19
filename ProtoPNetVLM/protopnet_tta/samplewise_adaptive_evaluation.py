#!/usr/bin/env python3
"""Evaluate fixed and samplewise adaptive ProtoTTA+ on SICAPv2-C."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = [0, 1, 2]
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform',
]
METHODS = {
    'FixedLambda0.2': 'ProtoHybrid',
    'SamplewiseAdaptive': 'ProtoSampleAdaptive',
    'SamplewiseAdaptiveGradNorm': 'ProtoSampleAdaptiveGradNorm',
    'SourceFreeRelative': 'ProtoSampleAdaptiveRelative',
    'SourceFreeRelativeGradNorm': 'ProtoSampleAdaptiveRelativeGradNorm',
    'TeacherMedian': 'ProtoSampleAdaptiveTeacherMedian',
    'TeacherBatchMedian': 'ProtoSampleAdaptiveTeacherBatchMedian',
    'CoverageRouter': 'ProtoSampleAdaptiveCoverageRouter',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate samplewise adaptive ProtoTTA+ on SICAPv2-C'
    )
    parser.add_argument(
        '--model',
        default='./saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth',
    )
    parser.add_argument('--data-dir', default='./datasets/SICAPv2_c')
    parser.add_argument(
        '--clean-data-dir',
        default='./datasets/SICAPv2_cropped/test_cropped',
    )
    parser.add_argument(
        '--output-dir',
        default='./samplewise_adaptive_sicapv2c',
    )
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpuid', default='0')
    parser.add_argument('--delta0', type=float, default=0.25)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--skip-prototype-metrics', action='store_true')
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def jobs():
    return [
        (method, mode, seed)
        for seed in SEEDS
        for method, mode in METHODS.items()
    ]


def result_path(output_dir, method, seed):
    return output_dir / 'raw' / f'{method}_seed_{seed}.json'


def evaluator_command(args, mode, seed, output, corruptions):
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
        '--lambda-proto', '0.2',
        '--adaptive-delta0', str(args.delta0),
        '--adaptive-top-k', str(args.top_k),
        '--track-efficiency',
    ]
    if not args.skip_prototype_metrics:
        command.append('--prototype-metrics')
    else:
        command.append('--skip-clean-evaluation')
    return command


def run_job(args, method, mode, seed, output_dir, corruptions=CORRUPTIONS):
    output = result_path(output_dir, method, seed)
    if corruptions != CORRUPTIONS:
        output = output_dir / 'smoke' / f'{method}_seed_{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = evaluator_command(
        args, mode, seed, output, corruptions
    )
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )
    return output


def pooled_mean_std(entries):
    total_count = sum(count for _, _, count in entries)
    if total_count == 0:
        return None, None
    mean = sum(value * count for value, _, count in entries) / total_count
    second_moment = sum(
        (std ** 2 + value ** 2) * count
        for value, std, count in entries
    ) / total_count
    variance = max(0.0, second_moment - mean ** 2)
    return mean, variance ** 0.5


def read_run(path, mode, severity):
    with path.open() as handle:
        payload = json.load(handle)
    results = payload['results'][mode]
    accuracies = {}
    lambda_entries = []
    adapted_samples = 0
    total_samples = 0
    proto_loss_sum = 0.0
    output_loss_sum = 0.0
    loss_sample_count = 0
    proto_gradient_sum = 0.0
    output_gradient_sum = 0.0
    gradient_batch_count = 0
    native_batches = 0
    router_batches = 0
    q_proto_sum = 0.0
    q_out_sum = 0.0
    reliable_fraction_sum = 0.0
    router_stat_batches = 0
    for corruption in CORRUPTIONS:
        result = results[corruption][str(severity)]
        accuracies[corruption] = 100.0 * result['accuracy']
        stats = result.get('adaptation_stats', {})
        adaptive_lambda = stats.get('adaptive_lambda', {})
        lambda_count = adaptive_lambda.get('sample_count', 0)
        if lambda_count:
            lambda_entries.append((
                adaptive_lambda['mean'],
                adaptive_lambda['std'],
                lambda_count,
            ))
        adapted_samples += stats.get('adapted_samples', 0)
        total_samples += stats.get('total_samples', 0)
        sample_count = adaptive_lambda.get('sample_count', 0)
        if sample_count:
            proto_loss_sum += stats['mean_proto_loss'] * sample_count
            output_loss_sum += stats['mean_output_loss'] * sample_count
            loss_sample_count += sample_count
        batch_count = stats.get('component_gradient_batch_count', 0)
        if batch_count:
            proto_gradient_sum += (
                stats['mean_proto_gradient_norm'] * batch_count
            )
            output_gradient_sum += (
                stats['mean_output_gradient_norm'] * batch_count
            )
            gradient_batch_count += batch_count
        router = stats.get('coverage_router') or {}
        total_router_batches = router.get('total_batches') or 0
        if total_router_batches and router.get('enabled'):
            native_fraction = router.get('native_batch_fraction') or 0.0
            native_batches += native_fraction * total_router_batches
            router_batches += total_router_batches
            q_proto_sum += (
                (router.get('mean_q_proto') or 0.0) * total_router_batches
            )
            q_out_sum += (
                (router.get('mean_q_out') or 0.0) * total_router_batches
            )
            reliable_fraction_sum += (
                (router.get('mean_reliable_fraction') or 0.0)
                * total_router_batches
            )
            router_stat_batches += total_router_batches
    lambda_mean, lambda_std = pooled_mean_std(lambda_entries)
    return {
        'per_corruption_percent': accuracies,
        'total_percent': statistics.fmean(accuracies.values()),
        'lambda_mean': lambda_mean,
        'lambda_std': lambda_std,
        'accepted_sample_percentage': (
            100.0 * adapted_samples / total_samples if total_samples else None
        ),
        'mean_proto_loss': (
            proto_loss_sum / loss_sample_count if loss_sample_count else None
        ),
        'mean_output_loss': (
            output_loss_sum / loss_sample_count if loss_sample_count else None
        ),
        'mean_proto_gradient_norm': (
            proto_gradient_sum / gradient_batch_count
            if gradient_batch_count else None
        ),
        'mean_output_gradient_norm': (
            output_gradient_sum / gradient_batch_count
            if gradient_batch_count else None
        ),
        'native_batch_fraction': (
            native_batches / router_batches if router_batches else None
        ),
        'mean_q_proto': (
            q_proto_sum / router_stat_batches
            if router_stat_batches else None
        ),
        'mean_q_out': (
            q_out_sum / router_stat_batches
            if router_stat_batches else None
        ),
        'mean_reliable_fraction': (
            reliable_fraction_sum / router_stat_batches
            if router_stat_batches else None
        ),
    }


def summarize(args, output_dir):
    method_runs = {
        method: {
            seed: read_run(
                result_path(output_dir, method, seed),
                mode,
                args.severity,
            )
            for seed in SEEDS
        }
        for method, mode in METHODS.items()
        if all(
            result_path(output_dir, method, seed).exists()
            for seed in SEEDS
        )
    }
    summary = {
        'provenance': {
            'model': args.model,
            'data_dir': args.data_dir,
            'clean_data_dir': args.clean_data_dir,
            'severity': args.severity,
            'batch_size': args.batch_size,
            'seeds': SEEDS,
            'corruptions': CORRUPTIONS,
            'fixed_lambda': 0.2,
            'adaptive_delta0': args.delta0,
            'adaptive_top_k': args.top_k,
            'batchwise_adaptive_lambda': 'not implemented in codebase',
            'selection_note': (
                'delta0 and top_k are label-free prespecified values; '
                'fixed lambda 0.2 came from the diagnostic test-set sweep.'
            ),
        },
        'methods': {},
    }
    fixed_mean = statistics.fmean(
        run['total_percent']
        for run in method_runs['FixedLambda0.2'].values()
    )
    for method, seed_runs in method_runs.items():
        totals = {
            str(seed): run['total_percent']
            for seed, run in seed_runs.items()
        }
        lambda_means = [
            run['lambda_mean']
            for run in seed_runs.values()
            if run['lambda_mean'] is not None
        ]
        lambda_stds = [
            run['lambda_std']
            for run in seed_runs.values()
            if run['lambda_std'] is not None
        ]
        mean_total = statistics.fmean(totals.values())
        summary['methods'][method] = {
            'mode': METHODS[method],
            'mean_total_percent': mean_total,
            'std_total_percent': statistics.stdev(totals.values()),
            'raw_seed_totals_percent': totals,
            'delta_vs_fixed_percent_points': mean_total - fixed_mean,
            'per_corruption_mean_percent': {
                corruption: statistics.fmean(
                    run['per_corruption_percent'][corruption]
                    for run in seed_runs.values()
                )
                for corruption in CORRUPTIONS
            },
            'per_corruption_std_percent': {
                corruption: statistics.stdev(
                    run['per_corruption_percent'][corruption]
                    for run in seed_runs.values()
                )
                for corruption in CORRUPTIONS
            },
            'mean_lambda': (
                statistics.fmean(lambda_means) if lambda_means else None
            ),
            'mean_within_run_lambda_std': (
                statistics.fmean(lambda_stds) if lambda_stds else None
            ),
            'mean_accepted_sample_percentage': statistics.fmean(
                run['accepted_sample_percentage']
                for run in seed_runs.values()
                if run['accepted_sample_percentage'] is not None
            ),
            'mean_proto_loss': (
                statistics.fmean(
                    run['mean_proto_loss']
                    for run in seed_runs.values()
                    if run['mean_proto_loss'] is not None
                ) if lambda_means else None
            ),
            'mean_output_loss': (
                statistics.fmean(
                    run['mean_output_loss']
                    for run in seed_runs.values()
                    if run['mean_output_loss'] is not None
                ) if lambda_means else None
            ),
            'mean_proto_gradient_norm': (
                statistics.fmean(
                    run['mean_proto_gradient_norm']
                    for run in seed_runs.values()
                    if run['mean_proto_gradient_norm'] is not None
                ) if lambda_means else None
            ),
            'mean_output_gradient_norm': (
                statistics.fmean(
                    run['mean_output_gradient_norm']
                    for run in seed_runs.values()
                    if run['mean_output_gradient_norm'] is not None
                ) if lambda_means else None
            ),
            'native_batch_fraction': (
                statistics.fmean(
                    run['native_batch_fraction']
                    for run in seed_runs.values()
                    if run['native_batch_fraction'] is not None
                ) if any(
                    run['native_batch_fraction'] is not None
                    for run in seed_runs.values()
                ) else None
            ),
            'mean_q_proto': (
                statistics.fmean(
                    run['mean_q_proto']
                    for run in seed_runs.values()
                    if run['mean_q_proto'] is not None
                ) if any(
                    run['mean_q_proto'] is not None
                    for run in seed_runs.values()
                ) else None
            ),
            'mean_q_out': (
                statistics.fmean(
                    run['mean_q_out']
                    for run in seed_runs.values()
                    if run['mean_q_out'] is not None
                ) if any(
                    run['mean_q_out'] is not None
                    for run in seed_runs.values()
                ) else None
            ),
            'mean_reliable_fraction': (
                statistics.fmean(
                    run['mean_reliable_fraction']
                    for run in seed_runs.values()
                    if run['mean_reliable_fraction'] is not None
                ) if any(
                    run['mean_reliable_fraction'] is not None
                    for run in seed_runs.values()
                ) else None
            ),
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
            raise ValueError(
                f'--task-id must be between 0 and {len(all_jobs) - 1}'
            )
        method, mode, seed = all_jobs[args.task_id]
        run_job(args, method, mode, seed, output_dir)
        return

    for method, mode, seed in all_jobs:
        run_job(args, method, mode, seed, output_dir)
    summarize(args, output_dir)


if __name__ == '__main__':
    main()
