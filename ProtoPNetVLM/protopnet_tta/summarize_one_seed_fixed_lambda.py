#!/usr/bin/env python3
"""Validate and summarize a one-seed fixed-lambda ProtoPNet sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CORRUPTIONS = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lambdas', type=float, nargs='+', required=True)
    parser.add_argument('--severity', type=int, default=5)
    return parser.parse_args()


def lambda_label(value: float) -> str:
    return f'{value:.1f}'


def load_strict(path: Path):
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f'{path}: non-finite JSON constant {value}')
        ),
    )


def validate_and_summarize(path: Path, seed: int, requested: float, severity: int):
    payload = load_strict(path)
    metadata = payload['metadata']
    if metadata['seed'] != seed:
        raise RuntimeError(f'{path}: seed {metadata["seed"]} != {seed}')
    if metadata['modes'] != ['ProtoHybrid']:
        raise RuntimeError(f'{path}: expected only ProtoHybrid mode')
    if metadata['corruption_types'] != list(CORRUPTIONS):
        raise RuntimeError(f'{path}: corruption order/set does not match protocol')

    mode_config = metadata['mode_configs']['ProtoHybrid']
    loss_definition = metadata['loss_definition']
    adaptive = metadata['samplewise_adaptive_lambda']
    if mode_config['alpha_proto'] != requested:
        raise RuntimeError(f'{path}: mode alpha_proto is not exactly {requested}')
    if mode_config['alpha_softmax'] != 1.0 - requested:
        raise RuntimeError(f'{path}: mode alpha_softmax mismatch')
    if loss_definition['lambda_proto'] != requested:
        raise RuntimeError(f'{path}: recorded loss lambda is not exactly {requested}')
    if loss_definition['lambda_output'] != 1.0 - requested:
        raise RuntimeError(f'{path}: recorded output lambda mismatch')
    if adaptive['enabled'] or adaptive['controllers']:
        raise RuntimeError(f'{path}: adaptive/samplewise controller was enabled')
    if adaptive['component_gradient_normalization']:
        raise RuntimeError(f'{path}: component gradient normalization was enabled')

    command = metadata['runtime']['command']
    forbidden_flags = {
        '--samplewise-adaptive-lambda',
        '--component-gradient-normalization',
        '--adaptive-controller',
    }
    present_forbidden = sorted(forbidden_flags.intersection(command))
    if present_forbidden:
        raise RuntimeError(f'{path}: forbidden flags present: {present_forbidden}')

    mode_results = payload['results']['ProtoHybrid']
    if set(mode_results) != set(CORRUPTIONS):
        raise RuntimeError(f'{path}: missing or extra corruption results')
    per_corruption = {}
    accepted_total = 0
    recorded_total = 0
    for corruption in CORRUPTIONS:
        result = mode_results[corruption][str(severity)]
        accuracy = result.get('accuracy')
        if accuracy is None:
            raise RuntimeError(f'{path}: {corruption} has no accuracy')
        per_corruption[corruption] = 100.0 * accuracy

        stats = result.get('adaptation_stats')
        if stats is None:
            raise RuntimeError(f'{path}: {corruption} has no adaptation stats')
        audit = stats.get('fixed_lambda_audit')
        if audit is None:
            raise RuntimeError(f'{path}: {corruption} has no fixed-lambda audit')
        accepted = int(stats['adapted_samples'])
        if audit['prototype_lambda'] != requested:
            raise RuntimeError(f'{path}: {corruption} audit lambda mismatch')
        if audit['output_lambda'] != 1.0 - requested:
            raise RuntimeError(f'{path}: {corruption} audit output lambda mismatch')
        if audit['distinct_prototype_lambda_values'] != [requested]:
            raise RuntimeError(f'{path}: {corruption} lambda was not constant')
        if not audit['constant_for_every_accepted_sample']:
            raise RuntimeError(f'{path}: {corruption} constant-lambda check failed')
        if audit['accepted_sample_count'] != accepted:
            raise RuntimeError(f'{path}: {corruption} accepted count mismatch')
        if audit['recorded_lambda_count'] != accepted:
            raise RuntimeError(f'{path}: {corruption} recorded count mismatch')
        accepted_total += accepted
        recorded_total += audit['recorded_lambda_count']

    values = list(per_corruption.values())
    return {
        'source_file': str(path.resolve()),
        'seed': seed,
        'requested_lambda': requested,
        'per_corruption_accuracy_percent': per_corruption,
        'across_corruptions': {
            'count': len(values),
            'mean_accuracy_percent': statistics.fmean(values),
            'population_std_accuracy_percent': statistics.pstdev(values),
            'minimum_accuracy_percent': min(values),
            'maximum_accuracy_percent': max(values),
        },
        'fixed_lambda_validation': {
            'passed': True,
            'adaptive_samplewise_router_logic_disabled': True,
            'component_gradient_normalization_disabled': True,
            'accepted_sample_count': accepted_total,
            'recorded_lambda_count': recorded_total,
            'distinct_recorded_lambda_values': [requested],
            'constant_and_exact_for_every_accepted_sample': True,
        },
    }


def main():
    args = parse_args()
    summary_path = args.output_dir / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {summary_path}')
    results = {}
    for requested in args.lambdas:
        label = lambda_label(requested)
        path = args.output_dir / 'raw' / f'lambda_{label}_seed_{args.seed}.json'
        results[label] = validate_and_summarize(
            path, args.seed, requested, args.severity
        )
    summary = {
        'protocol': {
            'backbone': 'ProtoPNet (BN + 1x1 convolutions)',
            'dataset': 'SICAPv2-C rebuilt locally with generation seed 0',
            'evaluation_seed': args.seed,
            'severity': args.severity,
            'corruptions': list(CORRUPTIONS),
            'lambdas': args.lambdas,
            'mode': 'ProtoHybrid fixed scalar lambda',
            'population_std_definition': (
                'statistics.pstdev over the 13 per-corruption accuracies'
            ),
        },
        'validation': {
            'status': 'passed',
            'result_file_count': len(results),
            'all_lambdas_exact_for_every_accepted_sample': True,
            'no_adaptive_or_router_logic': True,
            'no_component_gradient_normalization': True,
        },
        'lambda_results': results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Summary saved to {summary_path}', flush=True)


if __name__ == '__main__':
    main()
