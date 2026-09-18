#!/usr/bin/env python3
"""Validate and aggregate the corrected seeded-random ProtoPNet Table 3 run."""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import torch


SEEDS = (0, 2, 3)
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
STREAM_ORDER = 'seeded_random'
STREAM_ALGORITHM = (
    'torch_randperm_after_dataloader_base_seed_legacy_compatible'
)
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
MANUSCRIPT_ACCURACY = {
    'Unadapted': 31.4, 'MEMO': 55.9, 'SAR': 55.8, 'Tent': 54.3,
    'EATA': 55.7, 'ProtoTTA_final_adaptive_router': 56.2,
}


def sha256_lines(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def fixed_stream_indices(num_samples, stream_seed):
    generator = torch.Generator()
    generator.manual_seed(stream_seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.randperm(num_samples, generator=generator).tolist()


def controlled_identity(stream_seed):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    canonical_ids = sorted(
        str(path.relative_to(CLEAN)) for path in CLEAN.rglob('*')
        if path.is_file() and path.suffix.lower() in extensions
    )
    classes = sorted({Path(sample_id).parts[0] for sample_id in canonical_ids})
    class_to_idx = {name: index for index, name in enumerate(classes)}
    canonical_labels = [
        class_to_idx[Path(sample_id).parts[0]] for sample_id in canonical_ids
    ]
    indices = fixed_stream_indices(len(canonical_ids), stream_seed)
    ids = [canonical_ids[index] for index in indices]
    labels = [canonical_labels[index] for index in indices]
    return {
        'num_samples': len(ids),
        'ids': ids,
        'labels': labels,
        'sample_id_hash_sha256': sha256_lines(ids),
        'label_hash_sha256': sha256_lines(labels),
        'canonical_sample_id_hash_sha256': sha256_lines(canonical_ids),
        'stream_order_index_hash_sha256': sha256_lines(indices),
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
        raise RuntimeError(f'Missing result {mode}/{corruption}/5') from exc
    if not isinstance(result, dict):
        raise RuntimeError(f'Non-dictionary result {mode}/{corruption}/5')
    return result


def close(a, b, tolerance=1e-7):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def table_path(run_dir, label, seed):
    return run_dir / 'table3' / 'raw' / f'{label}_seed_{seed}.json'


def validate_payload(path, payload, seed, mode, identity):
    cache_path = path.parents[2] / 'clean_reference' / f'seed_{seed}.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    clean = cache['trajectory']
    clean_predictions = clean['predictions'].tolist()
    clean_labels = clean['labels'].tolist()
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
        'paired_num_samples': identity['num_samples'],
        'sample_id_hash_sha256': identity['sample_id_hash_sha256'],
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
    expected_protocol = {
        'shuffle': True,
        'stream_order': STREAM_ORDER,
        'stream_order_seed': seed,
        'stream_order_algorithm': STREAM_ALGORITHM,
        'stream_order_index_hash_sha256': identity[
            'stream_order_index_hash_sha256'
        ],
        'controlled_cohort_name': COHORT,
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            raise RuntimeError(
                f'{path}: protocol {field}={protocol.get(field)!r}, '
                f'expected {expected!r}'
            )

    required = (
        'accuracy', 'PAC_mean', 'PCA_weighted_mean',
        'calibration_agreement', 'efficiency', 'selection_rate',
        'paired_num_samples', 'clean_accuracy_reference',
        'stability_lower_bound', 'stability_upper_bound',
        'stability_bounds_passed', 'seed', 'method_config',
        'sample_id_hash_sha256', 'label_hash_sha256',
        'canonical_sample_id_hash_sha256', 'stream_order',
        'stream_order_seed', 'stream_order_algorithm',
        'stream_order_index_hash_sha256', 'model_checksum_sha256',
        'model_path', 'data_path', 'git_commit',
        'evaluator_checksum_sha256', 'evaluator_diff_sha256',
        'online_predictions', 'online_labels',
        'online_prediction_hash_sha256', 'clean_prediction_hash_sha256',
        'online_logits_hash_sha256',
        'online_prototype_activations_hash_sha256',
    )
    provenance = set()
    for corruption in CORRUPTIONS:
        result = result_at(payload, mode, corruption)
        missing = [field for field in required if field not in result]
        if missing:
            raise RuntimeError(f'{path}: {corruption} missing {missing}')
        for field in ('PAC_mean', 'PCA_weighted_mean', 'efficiency'):
            if result[field] is None:
                raise RuntimeError(f'{path}: {corruption} null {field}')
        identity_checks = {
            'paired_num_samples': identity['num_samples'],
            'sample_id_hash_sha256': identity['sample_id_hash_sha256'],
            'label_hash_sha256': identity['label_hash_sha256'],
            'canonical_sample_id_hash_sha256': identity[
                'canonical_sample_id_hash_sha256'
            ],
            'stream_order': STREAM_ORDER,
            'stream_order_seed': seed,
            'stream_order_algorithm': STREAM_ALGORITHM,
            'stream_order_index_hash_sha256': identity[
                'stream_order_index_hash_sha256'
            ],
            'seed': seed,
        }
        for field, expected in identity_checks.items():
            if result.get(field) != expected:
                raise RuntimeError(
                    f'{path}: {corruption} {field} mismatch'
                )
        predictions = result['online_predictions']
        labels = result['online_labels']
        if labels != identity['labels'] or clean_labels != identity['labels']:
            raise RuntimeError(f'{path}: {corruption} label/order mismatch')
        if len(predictions) != identity['num_samples']:
            raise RuntimeError(f'{path}: {corruption} prediction N mismatch')
        accuracy = sum(
            prediction == label
            for prediction, label in zip(predictions, labels)
        ) / identity['num_samples']
        clean_accuracy = sum(
            prediction == label
            for prediction, label in zip(clean_predictions, clean_labels)
        ) / identity['num_samples']
        stability = sum(
            prediction == clean_prediction
            for prediction, clean_prediction in zip(
                predictions, clean_predictions
            )
        ) / identity['num_samples']
        lower = max(0.0, clean_accuracy + accuracy - 1.0)
        upper = 1.0 - abs(clean_accuracy - accuracy)
        passed = lower - 1e-12 <= stability <= upper + 1e-12
        if not all((
            close(result['accuracy'], accuracy),
            close(result['clean_accuracy_reference'], clean_accuracy),
            close(result['calibration_agreement'], stability),
            close(result['stability_lower_bound'], lower),
            close(result['stability_upper_bound'], upper),
            result['stability_bounds_passed'] is True,
            passed,
        )):
            raise RuntimeError(
                f'{path}: {corruption} trajectory/bound validation failed'
            )
        expected_data = DATA / corruption / '5'
        if Path(result['data_path']).resolve() != expected_data:
            raise RuntimeError(f'{path}: {corruption} wrong data path')
        provenance.add((
            result['model_checksum_sha256'],
            result['evaluator_checksum_sha256'],
            result['evaluator_diff_sha256'], result['git_commit'],
        ))
    if len(provenance) != 1:
        raise RuntimeError(f'{path}: mixed provenance within file')
    return next(iter(provenance))


def summarize(values):
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
    identities = {seed: controlled_identity(seed) for seed in SEEDS}
    payloads = {}
    provenance = set()
    for label, mode in METHODS.items():
        for seed in SEEDS:
            path = table_path(run_dir, label, seed)
            payload = load_json(path)
            provenance.add(validate_payload(
                path, payload, seed, mode, identities[seed]
            ))
            payloads[(label, seed)] = payload
    if len(provenance) != 1:
        raise RuntimeError('Table 3 files mix evaluator/model provenance')

    rows = {}
    for label, mode in METHODS.items():
        metrics = {name: {c: [] for c in CORRUPTIONS} for name in (
            'accuracy', 'PAC', 'PCA-W', 'stability', 'selection_rate',
            'relative_speed'
        )}
        for seed in SEEDS:
            payload = payloads[(label, seed)]
            normal = payloads[('Unadapted', seed)]
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
        rows[label] = {name: summarize(values) for name, values in metrics.items()}

    comparison = {'table1_accuracy_canary': {}, 'table3': {}}
    for label, old in MANUSCRIPT_ACCURACY.items():
        corrected = rows[label]['accuracy']['mean']
        comparison['table1_accuracy_canary'][label] = {
            'manuscript': old, 'rerun': corrected,
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
    return {
        'status': 'passed',
        'aggregation_order': (
            'mean over seeds for each corruption, then mean and sample SD '
            'over 13 corruption settings'
        ),
        'seeds': list(SEEDS),
        'corruptions': list(CORRUPTIONS),
        'stream_protocol': {
            'order': STREAM_ORDER, 'seeds': list(SEEDS),
            'algorithm': STREAM_ALGORITHM,
            'index_hash_sha256_by_seed': {
                str(seed): identities[seed]['stream_order_index_hash_sha256']
                for seed in SEEDS
            },
        },
        'sample_identity_by_seed': identities,
        'provenance': list(provenance)[0],
        'table3_rows_percent': rows,
        'comparison_to_current_manuscript': comparison,
    }


def markdown_report(summary):
    lines = [
        '# Corrected ProtoPNet Table 3: seeded-random paired protocol', '',
        f"Validation: **{summary['status']}**", '',
        'Values are `mean ± sample SD` after seed averaging.', '',
        '| Method | PAC | PCA-W | Stability | Selection | Relative speed |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for label, row in summary['table3_rows_percent'].items():
        values = [row[key] for key in (
            'PAC', 'PCA-W', 'stability', 'selection_rate', 'relative_speed'
        )]
        rendered = [
            f"{value['mean']:.2f} ± {value['sample_sd']:.2f}"
            for value in values
        ]
        lines.append(f"| {label} | " + ' | '.join(rendered) + ' |')
    lines.extend([
        '', 'The aggregation also records accuracy deltas against Table 1 as '
        'a reproduction check. No manuscript files were modified.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    aggregate_path = run_dir / 'aggregate.json'
    report_path = run_dir / 'comparison.md'
    if aggregate_path.exists() or report_path.exists():
        raise FileExistsError('Refusing to overwrite existing aggregation')
    summary = validate_and_aggregate(run_dir)
    aggregate_path.write_text(json.dumps(summary, indent=2) + '\n')
    report_path.write_text(markdown_report(summary))
    print(json.dumps({
        'status': 'passed', 'aggregate': str(aggregate_path),
        'report': str(report_path),
    }))


if __name__ == '__main__':
    main()
