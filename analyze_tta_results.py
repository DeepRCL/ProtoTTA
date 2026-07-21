#!/usr/bin/env python3
"""Summarize robustness JSONs and adaptation diagnostics without labels/tuning."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path):
    with path.open() as handle:
        payload = json.load(handle)
    rows = []
    for mode, corruptions in payload.get('results', {}).items():
        for corruption, severities in corruptions.items():
            for severity, result in severities.items():
                if not isinstance(result, dict) or result.get('accuracy') is None:
                    continue
                rows.append((mode, corruption, str(severity), result))
    return payload, rows


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 'n/a'
    return f'{values.mean():.6g} ± {values.std(ddof=0):.6g} [{values.min():.6g}, {values.max():.6g}]'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_files', nargs='+', type=Path)
    parser.add_argument('--compare', nargs=2, metavar=('METHOD_A', 'METHOD_B'))
    parser.add_argument('--merge', action='store_true',
                        help='Treat all input JSON files as one result collection')
    args = parser.parse_args()

    collections = []
    if args.merge:
        rows = []
        for path in args.json_files:
            _, file_rows = load_rows(path)
            rows.extend(file_rows)
        collections.append(('merged inputs', rows))
    else:
        for path in args.json_files:
            _, rows = load_rows(path)
            collections.append((str(path), rows))

    for label, rows in collections:
        print(f'\n{label}')
        modes = sorted({row[0] for row in rows})
        by_key = {}
        for mode in modes:
            selected = [row for row in rows if row[0] == mode]
            accuracies = [row[3]['accuracy'] * 100.0 for row in selected]
            print(f'  {mode:<30} mean={np.mean(accuracies):7.3f}%  n={len(accuracies)}')
            for _, corruption, severity, result in selected:
                by_key[(mode, corruption, severity)] = result['accuracy'] * 100.0

            diagnostics = [
                row[3].get('adaptation_stats', {}) for row in selected
                if row[3].get('adaptation_stats')
            ]
            for key in (
                'proto_loss', 'output_loss', 'proto_grad_norm', 'output_grad_norm',
                'adaptive_lambda_raw', 'adaptive_lambda',
                'proto_signal_reliability', 'output_signal_reliability',
            ):
                values = [value for stats in diagnostics for value in stats.get(key, [])]
                if values:
                    print(f'    {key:<28} {describe(values)}')

        if args.compare:
            method_a, method_b = args.compare
            shared = sorted({
                (corruption, severity)
                for mode, corruption, severity in by_key if mode == method_a
            } & {
                (corruption, severity)
                for mode, corruption, severity in by_key if mode == method_b
            })
            if shared:
                deltas = np.asarray([
                    by_key[(method_a, corruption, severity)] -
                    by_key[(method_b, corruption, severity)]
                    for corruption, severity in shared
                ])
                print(
                    f'  paired {method_a} - {method_b}: '
                    f'{deltas.mean():+.3f} pp; wins/ties/losses='
                    f'{(deltas > 0).sum()}/{(deltas == 0).sum()}/{(deltas < 0).sum()}'
                )


if __name__ == '__main__':
    main()
