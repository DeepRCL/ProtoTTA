#!/usr/bin/env python3
"""Strict validation and aggregation for the corrected ProtoPNet-only runs."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch


SEEDS = (0, 2, 3)
LAMBDAS = (0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0)
CORRUPTIONS = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform',
)
METHODS = {
    'Unadapted': 'Normal',
    'MEMO': 'MEMO',
    'SAR': 'SAR',
    'Tent': 'Tent',
    'EATA': 'EATA',
    'ProtoTTA_final_adaptive_router':
        'ProtoAbsoluteConsistencyCoverageRouter',
}
COHORT = 'rebuilt_seed0_sicapv2c_controlled_four_seed'
MODEL = Path(os.environ.get(
    'PROTOPNET_MODEL',
    'saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth',
)).resolve()
DATA = Path(os.environ.get(
    'PROTOPNET_CORRUPTED_DATA',
    'datasets/SICAPv2_c_rebuilt_seed0',
)).resolve()
CLEAN = Path(os.environ.get(
    'PROTOPNET_CLEAN_DATA',
    'datasets/SICAPv2_cropped/test_cropped',
)).resolve()


def controlled_identity():
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    ids = sorted(
        str(path.relative_to(CLEAN)) for path in CLEAN.rglob('*')
        if path.is_file() and path.suffix.lower() in extensions
    )
    import hashlib
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode('utf-8'))
        digest.update(b'\n')
    return len(ids), digest.hexdigest()

MANUSCRIPT_TABLE3 = {
    'Unadapted': {'PAC': 88.8, 'PCA-W': 51.3, 'stability': 20.6,
                  'selection_rate': 0.0, 'relative_speed': 100.0},
    'MEMO': {'PAC': 86.2, 'PCA-W': 60.7, 'stability': 30.6,
             'selection_rate': 100.0, 'relative_speed': 45.5},
    'SAR': {'PAC': 86.0, 'PCA-W': 61.9, 'stability': 28.0,
            'selection_rate': 100.0, 'relative_speed': 22.6},
    'Tent': {'PAC': 84.5, 'PCA-W': 61.2, 'stability': 22.1,
             'selection_rate': 100.0, 'relative_speed': 45.5},
    'EATA': {'PAC': 86.2, 'PCA-W': 61.8, 'stability': 27.5,
             'selection_rate': 3.0, 'relative_speed': 89.3},
    'ProtoTTA_final_adaptive_router': {
        'PAC': 85.9, 'PCA-W': 62.4, 'stability': 29.4,
        'selection_rate': 78.1, 'relative_speed': 11.1,
    },
}
MANUSCRIPT_LAMBDAS = {
    '0.0': 55.97, '0.2': 56.02, '0.4': 56.17, '0.5': 56.16,
    '0.7': 55.84, '0.8': 54.36, '1.0': 51.70,
}
MANUSCRIPT_ACCURACY = {
    'Unadapted': 31.4, 'MEMO': 55.9, 'SAR': 55.8, 'Tent': 54.3,
    'EATA': 55.7, 'ProtoTTA_final_adaptive_router': 56.2,
}


def load_json(path):
    try:
        return json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f'non-finite JSON constant {value}')
            ),
        )
    except Exception as exc:
        raise RuntimeError(f'Cannot load {path}: {exc}') from exc


def result_at(payload, mode, corruption):
    try:
        result = payload['results'][mode][corruption]['5']
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f'Missing result {mode}/{corruption}/5'
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError(f'Non-dictionary result {mode}/{corruption}/5')
    return result


def close(a, b, tolerance=1e-7):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def validate_payload(path, payload, seed, mode, require_interpretability,
                     expected_lambda=None):
    controlled_n, controlled_id_hash = controlled_identity()
    cache_path = path.parents[2] / 'clean_reference' / f'seed_{seed}.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    clean_trajectory = cache['trajectory']
    clean_predictions = clean_trajectory['predictions'].tolist()
    clean_labels = clean_trajectory['labels'].tolist()
    metadata = payload.get('metadata', {})
    expected_metadata = {
        'seed': seed,
        'model_path': str(MODEL),
        'data_dir': str(DATA),
        'clean_data_dir': str(CLEAN),
        'severity': 5,
        'batch_size': 64,
        'corruption_types': list(CORRUPTIONS),
        'modes': [mode],
    }
    for field, expected in expected_metadata.items():
        actual = metadata.get(field)
        if field.endswith('_path') or field.endswith('_dir'):
            actual = str(Path(actual).resolve()) if actual else actual
        if actual != expected:
            raise RuntimeError(
                f'{path}: metadata {field}={actual!r}, expected {expected!r}'
            )
    protocol = metadata.get('protocol', {})
    if protocol.get('shuffle') is not False:
        raise RuntimeError(f'{path}: shuffle must be false')
    if protocol.get('controlled_cohort_name') != COHORT:
        raise RuntimeError(f'{path}: wrong/missing controlled cohort')
    if metadata.get('paired_num_samples') != controlled_n:
        raise RuntimeError(
            f'{path}: paired N does not match verified clean N={controlled_n}'
        )

    if expected_lambda is not None:
        config = metadata.get('mode_configs', {}).get(mode, {})
        if (not close(config.get('alpha_proto'), expected_lambda) or
                not close(config.get('alpha_softmax'), 1.0 - expected_lambda)):
            raise RuntimeError(f'{path}: fixed-lambda config mismatch')

    required = (
        'accuracy', 'PAC_mean', 'PCA_weighted_mean',
        'calibration_agreement', 'efficiency', 'selection_rate',
        'paired_num_samples', 'clean_accuracy_reference',
        'stability_lower_bound', 'stability_upper_bound',
        'stability_bounds_passed', 'seed', 'method_config',
        'sample_id_hash_sha256', 'model_checksum_sha256', 'model_path',
        'data_path', 'git_commit', 'evaluator_checksum_sha256',
        'evaluator_diff_sha256', 'online_predictions', 'online_labels',
        'online_prediction_hash_sha256', 'clean_prediction_hash_sha256',
        'online_logits_hash_sha256',
        'online_prototype_activations_hash_sha256',
    )
    identities = set()
    provenance = set()
    for corruption in CORRUPTIONS:
        result = result_at(payload, mode, corruption)
        missing = [field for field in required if field not in result]
        if missing:
            raise RuntimeError(f'{path}: {corruption} missing {missing}')
        if require_interpretability:
            for field in ('PAC_mean', 'PCA_weighted_mean', 'efficiency'):
                if result[field] is None:
                    raise RuntimeError(f'{path}: {corruption} null {field}')
        if (result['paired_num_samples'] != controlled_n or
                result['sample_id_hash_sha256'] != controlled_id_hash or
                result['seed'] != seed):
            raise RuntimeError(f'{path}: {corruption} N/seed mismatch')
        clean_accuracy = float(result['clean_accuracy_reference'])
        accuracy = float(result['accuracy'])
        stability = float(result['calibration_agreement'])
        predictions = result['online_predictions']
        labels = result['online_labels']
        if (len(predictions) != controlled_n or labels != clean_labels or
                len(clean_predictions) != controlled_n):
            raise RuntimeError(f'{path}: {corruption} trajectory mismatch')
        recomputed_accuracy = sum(
            prediction == label
            for prediction, label in zip(predictions, labels)
        ) / controlled_n
        recomputed_clean_accuracy = sum(
            prediction == label
            for prediction, label in zip(clean_predictions, clean_labels)
        ) / controlled_n
        recomputed_stability = sum(
            prediction == clean_prediction
            for prediction, clean_prediction in zip(
                predictions, clean_predictions
            )
        ) / controlled_n
        if (not close(accuracy, recomputed_accuracy) or
                not close(clean_accuracy, recomputed_clean_accuracy) or
                not close(stability, recomputed_stability)):
            raise RuntimeError(
                f'{path}: {corruption} saved metrics do not match exact '
                'trajectory predictions'
            )
        lower = max(0.0, clean_accuracy + accuracy - 1.0)
        upper = 1.0 - abs(clean_accuracy - accuracy)
        passed = lower - 1e-12 <= stability <= upper + 1e-12
        if (not close(result['stability_lower_bound'], lower) or
                not close(result['stability_upper_bound'], upper) or
                result['stability_bounds_passed'] is not True or not passed):
            raise RuntimeError(
                f'{path}: {corruption} stability validation failed: '
                f'{lower} <= {stability} <= {upper}'
            )
        identities.add((result['sample_id_hash_sha256'],
                        result.get('label_hash_sha256')))
        provenance.add((result['model_checksum_sha256'],
                        result['evaluator_checksum_sha256'],
                        result['evaluator_diff_sha256'], result['git_commit']))
        expected_data = DATA / corruption / '5'
        if Path(result['data_path']).resolve() != expected_data:
            raise RuntimeError(f'{path}: {corruption} wrong data path')
    if len(identities) != 1 or len(provenance) != 1:
        raise RuntimeError(f'{path}: mixed identity/provenance within file')
    return next(iter(identities)), next(iter(provenance))


def table_path(run_dir, label, seed):
    return run_dir / 'table3' / 'raw' / f'{label}_seed_{seed}.json'


def sweep_path(run_dir, value, seed):
    return run_dir / 'lambda_sweep' / 'raw' / (
        f'lambda_{value:.1f}_seed_{seed}.json'
    )


def seed_then_corruption_summary(values):
    corruption_means = {
        corruption: statistics.fmean(values[corruption])
        for corruption in CORRUPTIONS
    }
    vector = list(corruption_means.values())
    return {
        'per_corruption_seed_mean': corruption_means,
        'mean': statistics.fmean(vector),
        'sample_sd': statistics.stdev(vector),
    }


def validate_and_aggregate(run_dir):
    table_payloads = {}
    identities = set()
    provenance = set()
    for label, mode in METHODS.items():
        for seed in SEEDS:
            path = table_path(run_dir, label, seed)
            payload = load_json(path)
            identity, prov = validate_payload(
                path, payload, seed, mode, require_interpretability=True
            )
            identities.add(identity)
            provenance.add(prov)
            table_payloads[(label, seed)] = payload
    if len(identities) != 1 or len(provenance) != 1:
        raise RuntimeError('Table 3 files mix sample identity or provenance')

    rows = {}
    for label, mode in METHODS.items():
        metrics = {name: {c: [] for c in CORRUPTIONS} for name in (
            'accuracy', 'PAC', 'PCA-W', 'stability', 'selection_rate',
            'relative_speed'
        )}
        for seed in SEEDS:
            payload = table_payloads[(label, seed)]
            normal = table_payloads[('Unadapted', seed)]
            for corruption in CORRUPTIONS:
                result = result_at(payload, mode, corruption)
                baseline = result_at(normal, 'Normal', corruption)
                metrics['accuracy'][corruption].append(result['accuracy'] * 100)
                metrics['PAC'][corruption].append(result['PAC_mean'] * 100)
                metrics['PCA-W'][corruption].append(
                    result['PCA_weighted_mean'] * 100
                )
                metrics['stability'][corruption].append(
                    result['calibration_agreement'] * 100
                )
                metrics['selection_rate'][corruption].append(
                    result['selection_rate'] * 100
                )
                speed = 100.0 if label == 'Unadapted' else 100.0 * (
                    result['efficiency']['throughput_samples_per_sec'] /
                    baseline['efficiency']['throughput_samples_per_sec']
                )
                metrics['relative_speed'][corruption].append(speed)
        rows[label] = {
            name: seed_then_corruption_summary(values)
            for name, values in metrics.items()
        }

    lambdas = {}
    for value in LAMBDAS:
        values = {c: [] for c in CORRUPTIONS}
        for seed in SEEDS:
            path = sweep_path(run_dir, value, seed)
            payload = load_json(path)
            identity, prov = validate_payload(
                path, payload, seed, 'ProtoHybrid',
                require_interpretability=False, expected_lambda=value
            )
            identities.add(identity)
            provenance.add(prov)
            for corruption in CORRUPTIONS:
                values[corruption].append(
                    result_at(payload, 'ProtoHybrid', corruption)['accuracy']
                    * 100
                )
        lambdas[f'{value:.1f}'] = seed_then_corruption_summary(values)
    if len(identities) != 1 or len(provenance) != 1:
        raise RuntimeError('Full run mixes sample identity or provenance')

    comparison = {
        'table1_accuracy': {}, 'table3': {}, 'lambda_sweep': {}
    }
    for label, old in MANUSCRIPT_ACCURACY.items():
        corrected = rows[label]['accuracy']['mean']
        comparison['table1_accuracy'][label] = {
            'manuscript': old,
            'corrected': corrected,
            'delta': corrected - old,
        }
    for label, current in MANUSCRIPT_TABLE3.items():
        comparison['table3'][label] = {
            metric: {
                'manuscript': old,
                'corrected': rows[label][metric]['mean'],
                'delta': rows[label][metric]['mean'] - old,
            }
            for metric, old in current.items()
        }
    for label, old in MANUSCRIPT_LAMBDAS.items():
        comparison['lambda_sweep'][label] = {
            'manuscript': old,
            'corrected': lambdas[label]['mean'],
            'delta': lambdas[label]['mean'] - old,
        }
    return {
        'status': 'passed',
        'aggregation_order': (
            'mean over seeds for each corruption, then mean and sample SD '
            'over 13 corruption settings'
        ),
        'seeds': list(SEEDS),
        'corruptions': list(CORRUPTIONS),
        'sample_identity': list(identities)[0],
        'provenance': list(provenance)[0],
        'table3_rows_percent': rows,
        'lambda_accuracy_percent': lambdas,
        'comparison_to_current_manuscript': comparison,
    }


def markdown_report(summary):
    lines = [
        '# Corrected ProtoPNet controlled results', '',
        f"Validation: **{summary['status']}**", '',
        'Table 3 values are `mean ± sample SD` after seed averaging.', '',
        '| Method | PAC | PCA-W | Stability | Selection | Relative speed |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for label, row in summary['table3_rows_percent'].items():
        values = [row[k] for k in (
            'PAC', 'PCA-W', 'stability', 'selection_rate', 'relative_speed'
        )]
        rendered = [f"{x['mean']:.2f} ± {x['sample_sd']:.2f}" for x in values]
        lines.append(f"| {label} | " + ' | '.join(rendered) + ' |')
    lines.extend(['', '| Lambda | Mean accuracy | Sample SD | Manuscript | Delta |',
                  '|---:|---:|---:|---:|---:|'])
    for label, result in summary['lambda_accuracy_percent'].items():
        comparison = summary['comparison_to_current_manuscript'][
            'lambda_sweep'
        ][label]
        lines.append(
            f"| {label} | {result['mean']:.3f} | {result['sample_sd']:.3f} "
            f"| {comparison['manuscript']:.2f} | {comparison['delta']:+.3f} |"
        )
    lines.extend(['',
        'No manuscript files were modified. See `aggregate.json` for the full '
        'per-corruption results and manuscript comparison.'
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    aggregate_path = args.run_dir / 'aggregate.json'
    report_path = args.run_dir / 'comparison.md'
    if aggregate_path.exists() or report_path.exists():
        raise FileExistsError('Refusing to overwrite existing aggregation')
    summary = validate_and_aggregate(args.run_dir.resolve())
    aggregate_path.write_text(json.dumps(summary, indent=2) + '\n')
    report_path.write_text(markdown_report(summary))
    print(json.dumps({'status': 'passed', 'aggregate': str(aggregate_path),
                      'report': str(report_path)}))


if __name__ == '__main__':
    main()
