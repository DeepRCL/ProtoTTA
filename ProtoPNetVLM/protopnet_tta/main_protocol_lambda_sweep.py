#!/usr/bin/env python3
"""Audit and summarize the controlled-rebuild ProtoPNet lambda sweep."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path(os.environ.get('PROTOTTA_RESULTS_ROOT', '.')).resolve()
CANONICAL_ROOT = ROOT / 'protopnet_four_seed_rebuilt_sicapv2c/raw'
SEEDS = (0, 2, 3)
LAMBDAS = (0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0)
CORRUPTIONS = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform',
)
TABLE_DISPLAY_ORDER = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'defocus_blur', 'gaussian_blur', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
)
EXPECTED_TABLE_ROUNDED = {
    'gaussian_noise': 57.7, 'shot_noise': 58.0,
    'impulse_noise': 59.2, 'speckle_noise': 57.8,
    'defocus_blur': 53.8, 'gaussian_blur': 55.1,
    'frost': 60.0, 'fog': 51.0, 'brightness': 60.1,
    'contrast': 44.0, 'elastic_transform': 58.5,
    'jpeg_compression': 51.9, 'pixelate': 61.1,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'summarize'))
    parser.add_argument('--run-dir', type=Path, required=True)
    return parser.parse_args()


def load_strict(path: Path):
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f'{path}: non-finite JSON constant {value}')
        ),
    )


def result_at(payload, corruption):
    result = payload['results']['ProtoHybrid'][corruption]
    return result.get('5', result)


def canonical_path(seed):
    return CANONICAL_ROOT / f'ProtoTTA_fixed_lambda_0.2_seed_{seed}.json'


def lambda_label(value):
    return f'{value:.1f}'


def sweep_path(run_dir, value, seed):
    return run_dir / 'raw' / f'ProtoTTA_fixed_lambda_{lambda_label(value)}_seed_{seed}.json'


def canonical_payloads():
    payloads = {}
    for seed in SEEDS:
        path = canonical_path(seed)
        if not path.is_file():
            raise RuntimeError(f'Missing canonical result: {path}')
        payloads[seed] = load_strict(path)
    return payloads


def assert_canonical_protocol(payloads):
    reference = payloads[SEEDS[0]]['metadata']
    expected_command_tail = [
        '--severity', '5', '--batch_size', '64', '--seed', None,
        '--gpuid', '0', '--modes', 'ProtoHybrid', '--corruptions',
        *CORRUPTIONS, '--lambda-proto', '0.2', '--adaptive-delta0',
        '0.25', '--adaptive-top-k', '3', '--prototype-metrics',
        '--track-efficiency',
    ]
    protocol_fields = (
        'model_path', 'data_dir', 'clean_data_dir', 'severity', 'batch_size',
        'corruption_types', 'modes', 'optimizer', 'protocol',
        'prototype_metrics_enabled', 'efficiency_tracking_enabled',
    )
    for seed, payload in payloads.items():
        md = payload['metadata']
        if md['seed'] != seed:
            raise RuntimeError(f'Canonical seed mismatch in seed {seed}')
        for field in protocol_fields:
            if md[field] != reference[field]:
                raise RuntimeError(f'Canonical metadata mismatch for {field}')
        if md['mode_configs'] != {
            'ProtoHybrid': {
                'geo_filter_threshold': 0.8,
                'alpha_proto': 0.2,
                'alpha_softmax': 0.8,
            }
        }:
            raise RuntimeError(f'Canonical mode config mismatch for seed {seed}')
        loss = md['loss_definition']
        if loss['lambda_proto'] != 0.2 or loss['lambda_output'] != 0.8:
            raise RuntimeError(f'Canonical loss lambda mismatch for seed {seed}')
        adaptive = md['samplewise_adaptive_lambda']
        if adaptive['enabled'] or adaptive['controllers']:
            raise RuntimeError(f'Canonical fixed run has adaptive logic at seed {seed}')
        if adaptive['component_gradient_normalization']:
            raise RuntimeError(
                f'Canonical fixed run unexpectedly has gradient normalization at seed {seed}'
            )
        if list(payload['results']) != ['ProtoHybrid']:
            raise RuntimeError(f'Canonical seed {seed} contains unexpected modes')
        if set(payload['results']['ProtoHybrid']) != set(CORRUPTIONS):
            raise RuntimeError(f'Canonical seed {seed} has incomplete corruptions')
        for corruption in CORRUPTIONS:
            result = result_at(payload, corruption)
            for field in (
                'accuracy', 'PAC_mean', 'PCA_weighted_mean',
                'calibration_agreement', 'efficiency', 'adaptation_stats',
            ):
                if result.get(field) is None:
                    raise RuntimeError(
                        f'Canonical seed {seed}/{corruption} missing {field}'
                    )


def canonical_accuracy_audit(payloads):
    per_seed = {}
    per_corruption = {}
    for seed, payload in payloads.items():
        values = [result_at(payload, c)['accuracy'] * 100 for c in CORRUPTIONS]
        per_seed[str(seed)] = statistics.fmean(values)
    for corruption in TABLE_DISPLAY_ORDER:
        values = {
            str(seed): result_at(payloads[seed], corruption)['accuracy'] * 100
            for seed in SEEDS
        }
        mean = statistics.fmean(values.values())
        if round(mean, 1) != EXPECTED_TABLE_ROUNDED[corruption]:
            raise RuntimeError(
                f'Main-table mismatch for {corruption}: {mean:.8f} rounds to '
                f'{round(mean, 1)}, expected {EXPECTED_TABLE_ROUNDED[corruption]}'
            )
        per_corruption[corruption] = {
            'per_seed_accuracy_percent': values,
            'mean_over_seeds_percent': mean,
            'main_table_rounded_percent': EXPECTED_TABLE_ROUNDED[corruption],
        }
    means = [x['mean_over_seeds_percent'] for x in per_corruption.values()]
    overall = statistics.fmean(means)
    sample_std = statistics.stdev(means)
    if round(overall, 1) != 56.0 or round(sample_std, 1) != 4.8:
        raise RuntimeError(
            f'Main-table total mismatch: {overall:.8f} +/- {sample_std:.8f}'
        )
    return {
        'passed': True,
        'per_seed_accuracy_percent': per_seed,
        'per_corruption': per_corruption,
        'overall_mean_percent': overall,
        'sample_std_across_corruption_means_percent': sample_std,
        'main_table_display': '56.0 +/- 4.8',
    }


def preflight(run_dir):
    if (run_dir / 'preflight_audit.json').exists():
        raise FileExistsError('Refusing to overwrite preflight_audit.json')
    payloads = canonical_payloads()
    assert_canonical_protocol(payloads)
    accuracy = canonical_accuracy_audit(payloads)
    missing = [
        {'task_id': task_id, 'lambda': value, 'seed': seed,
         'output': str(sweep_path(run_dir, value, seed))}
        for task_id, (value, seed) in enumerate(
            (value, seed)
            for value in LAMBDAS if value != 0.2
            for seed in SEEDS
        )
    ]
    if len(missing) != 18:
        raise RuntimeError(f'Expected 18 missing jobs, got {len(missing)}')
    audit = {
        'status': 'passed',
        'canonical_method': 'ProtoTTA fixed lambda=0.2',
        'canonical_files': {
            str(seed): str(canonical_path(seed)) for seed in SEEDS
        },
        'canonical_metadata': copy.deepcopy(payloads[SEEDS[0]]['metadata']),
        'canonical_accuracy_reproduction': accuracy,
        'canonical_gradient_normalization': False,
        'note': (
            'The controlled-rebuild main-table fixed-lambda path is explicitly '
            'non-samplewise and records component_gradient_normalization=false. '
            'This value is preserved for protocol identity.'
        ),
        'requested_lambdas': list(LAMBDAS),
        'requested_seeds': list(SEEDS),
        'reused_lambda_0.2_jobs': 3,
        'missing_job_count': len(missing),
        'missing_jobs': missing,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'raw').mkdir(exist_ok=True)
    (run_dir / 'preflight_audit.json').write_text(
        json.dumps(audit, indent=2, allow_nan=False) + '\n'
    )
    (run_dir / 'missing_jobs.json').write_text(
        json.dumps(missing, indent=2, allow_nan=False) + '\n'
    )
    print(json.dumps({
        'status': 'passed', 'overall': accuracy['overall_mean_percent'],
        'std': accuracy['sample_std_across_corruption_means_percent'],
        'missing_jobs': len(missing),
    }))


def validate_sweep_payload(payload, canonical, requested, seed, path):
    md = payload['metadata']
    canonical_md = canonical['metadata']
    if md['seed'] != seed:
        raise RuntimeError(f'{path}: seed mismatch')
    invariant_fields = (
        'model_path', 'data_dir', 'clean_data_dir', 'severity', 'batch_size',
        'corruption_types', 'modes', 'optimizer', 'protocol',
        'prototype_metrics_enabled', 'proto_baseline_samples',
        'efficiency_tracking_enabled',
    )
    for field in invariant_fields:
        if md[field] != canonical_md[field]:
            raise RuntimeError(f'{path}: protocol field changed: {field}')
    if md['mode_configs']['ProtoHybrid'] != {
        'geo_filter_threshold': 0.8,
        'alpha_proto': requested,
        'alpha_softmax': 1.0 - requested,
    }:
        raise RuntimeError(f'{path}: wrong fixed-lambda mode config')
    if md['loss_definition']['lambda_proto'] != requested:
        raise RuntimeError(f'{path}: wrong loss lambda')
    adaptive = md['samplewise_adaptive_lambda']
    if adaptive['enabled'] or adaptive['controllers']:
        raise RuntimeError(f'{path}: adaptive/router path enabled')
    if adaptive['component_gradient_normalization']:
        raise RuntimeError(f'{path}: gradient normalization differs from canonical')
    if set(payload['results']['ProtoHybrid']) != set(CORRUPTIONS):
        raise RuntimeError(f'{path}: incomplete corruptions')
    accepted_total = 0
    for corruption in CORRUPTIONS:
        result = result_at(payload, corruption)
        for field in (
            'accuracy', 'PAC_mean', 'PCA_weighted_mean',
            'calibration_agreement', 'efficiency', 'adaptation_stats',
        ):
            if result.get(field) is None:
                raise RuntimeError(f'{path}: {corruption} missing {field}')
        stats = result['adaptation_stats']
        accepted = int(stats['adapted_samples'])
        audit = stats.get('fixed_lambda_audit')
        if audit is None:
            raise RuntimeError(f'{path}: {corruption} missing lambda audit')
        if audit['prototype_lambda'] != requested:
            raise RuntimeError(f'{path}: {corruption} wrong audited lambda')
        if audit['distinct_prototype_lambda_values'] != [requested]:
            raise RuntimeError(f'{path}: {corruption} nonconstant lambda')
        if audit['recorded_lambda_count'] != accepted:
            raise RuntimeError(f'{path}: {corruption} lambda count mismatch')
        if not audit['constant_for_every_accepted_sample']:
            raise RuntimeError(f'{path}: {corruption} lambda validation failed')
        accepted_total += accepted
    return accepted_total


def summarize(run_dir):
    summary_path = run_dir / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {summary_path}')
    canonical = canonical_payloads()
    assert_canonical_protocol(canonical)
    canonical_accuracy = canonical_accuracy_audit(canonical)
    lambda_results = {}
    completed_matrix = {}
    for value in LAMBDAS:
        label = lambda_label(value)
        seed_payloads = {}
        completed_matrix[label] = {}
        for seed in SEEDS:
            if value == 0.2:
                path = canonical_path(seed)
                source = 'reused_canonical_main_table_result'
            else:
                path = sweep_path(run_dir, value, seed)
                source = 'new_sweep_result'
                if not path.is_file():
                    raise RuntimeError(f'Missing sweep result: {path}')
            payload = load_strict(path)
            accepted = None
            if value != 0.2:
                accepted = validate_sweep_payload(
                    payload, canonical[seed], value, seed, path
                )
            seed_payloads[seed] = payload
            completed_matrix[label][str(seed)] = {
                'status': 'complete', 'source': source,
                'path': str(path), 'accepted_samples': accepted,
            }

        per_seed = {}
        for seed, payload in seed_payloads.items():
            per_seed[str(seed)] = statistics.fmean(
                result_at(payload, c)['accuracy'] * 100 for c in CORRUPTIONS
            )
        per_corruption = {}
        for corruption in TABLE_DISPLAY_ORDER:
            values = {
                str(seed): result_at(seed_payloads[seed], corruption)['accuracy'] * 100
                for seed in SEEDS
            }
            per_corruption[corruption] = {
                'per_seed_accuracy_percent': values,
                'mean_over_seeds_percent': statistics.fmean(values.values()),
            }
        corruption_means = [
            entry['mean_over_seeds_percent'] for entry in per_corruption.values()
        ]
        lambda_results[label] = {
            'lambda': value,
            'per_seed_mean_accuracy_percent': per_seed,
            'per_corruption': per_corruption,
            'overall_mean_across_seeds_and_corruptions_percent': (
                statistics.fmean(corruption_means)
            ),
            'population_std_across_corruption_means_percent': (
                statistics.pstdev(corruption_means)
            ),
            'sample_std_across_corruption_means_percent': (
                statistics.stdev(corruption_means)
            ),
            'minimum_corruption_mean_percent': min(corruption_means),
            'maximum_corruption_mean_percent': max(corruption_means),
        }
    summary = {
        'status': 'passed',
        'seeds': list(SEEDS),
        'user_facing_seeds': [1, 3, 4],
        'lambdas': list(LAMBDAS),
        'canonical_lambda_0.2_accuracy_reproduction': canonical_accuracy,
        'completed_missing_matrix': completed_matrix,
        'lambda_results': lambda_results,
        'protocol_validation': {
            'passed': True,
            'only_fixed_lambda_and_seed_differed_from_canonical': True,
            'output_path_also_changed_to_prevent_overwrite': True,
            'canonical_component_gradient_normalization': False,
            'adaptive_samplewise_router_logic_disabled': True,
            'all_new_results_contain_complete_metrics_and_lambda_audit': True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(summary_path)


def main():
    args = parse_args()
    if args.action == 'preflight':
        preflight(args.run_dir.resolve())
    else:
        summarize(args.run_dir.resolve())


if __name__ == '__main__':
    main()
