#!/usr/bin/env python3
"""Analyze the seeded, paper-compatible ProtoPFormer lambda sweep."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


LAMBDA_RE = re.compile(r'lambda_(.+)_seed(\d+)\.json$')
SEED_RE = re.compile(r'(tent|eata|adaptive)_seed(\d+)\.json$')


def read_accuracy(path, mode):
    with path.open() as handle:
        results = json.load(handle)['results'][mode]
    return {
        corruption: metrics['5']['accuracy'] * 100.0
        for corruption, metrics in results.items()
        if isinstance(metrics.get('5'), dict)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()

    lambda_runs = {}
    method_runs = {'tent': {}, 'eata': {}, 'adaptive': {}}
    for path in args.directory.glob('*.json'):
        match = LAMBDA_RE.match(path.name)
        if match:
            value, seed = float(match.group(1)), int(match.group(2))
            lambda_runs.setdefault(value, {})[seed] = read_accuracy(path, 'proto_tta')
            continue
        match = SEED_RE.match(path.name)
        if match:
            method, seed = match.group(1), int(match.group(2))
            mode = 'proto_tta_adaptive' if method == 'adaptive' else method
            method_runs[method][seed] = read_accuracy(path, mode)

    print('Paper-compatible model mode: train (upstream Tent behavior)')
    print('\nFixed lambda sweep (mean over corruptions per seed):')
    best = None
    for value in sorted(lambda_runs):
        seed_means = [np.mean(list(run.values())) for run in lambda_runs[value].values()]
        if not seed_means:
            continue
        mean, std = float(np.mean(seed_means)), float(np.std(seed_means))
        print(f'  lambda={value:>4.2f}: {mean:7.3f} +/- {std:.3f}  seeds={seed_means}')
        if best is None or mean > best[1]:
            best = (value, mean)
    if best:
        print(f'  descriptive best: lambda={best[0]:.2f} ({best[1]:.3f}%)')

    print('\nMethods:')
    for method, runs in method_runs.items():
        seed_means = [np.mean(list(run.values())) for run in runs.values()]
        if seed_means:
            print(f'  {method:<10}: {np.mean(seed_means):7.3f} +/- {np.std(seed_means):.3f}  seeds={seed_means}')

    adaptive = method_runs['adaptive']
    for baseline in ('tent', 'eata'):
        deltas = []
        for seed in sorted(set(adaptive) & set(method_runs[baseline])):
            shared = sorted(set(adaptive[seed]) & set(method_runs[baseline][seed]))
            deltas.extend(adaptive[seed][name] - method_runs[baseline][seed][name] for name in shared)
        if deltas:
            values = np.asarray(deltas)
            print(
                f'  adaptive - {baseline}: {values.mean():+.3f} pp; '
                f'wins/ties/losses={(values > 0).sum()}/{(values == 0).sum()}/{(values < 0).sum()}'
            )


if __name__ == '__main__':
    main()
