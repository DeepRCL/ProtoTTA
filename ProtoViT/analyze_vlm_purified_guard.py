#!/usr/bin/env python3
"""Aggregate the frozen three-seed confidence-guard confirmation."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

import vlm_eval
from semantic_prototype_guidance import write_json

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/fixed_lambda_metrics/10819"
RESULTS = ROOT / "results/semantic_prototta/guard_confirm"
SEEDS = [0, 1, 2]


def exact_sign_flip(values):
    observed = abs(float(values.mean()))
    null = [abs(float((values * np.asarray(signs)).mean()))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def main():
    per_corruption = {}
    base_seed_means, method_seed_means = [], []
    for seed in SEEDS:
        baseline = next(iter(json.loads(
            (BASE / f"cub200c_fixed_lambda1.0_seed{seed}.json").read_text()
        )["results"].values()))
        method = json.loads((RESULTS / f"seed{seed}.json").read_text())["results"]
        base_seed_means.append(np.mean([100 * baseline[c]["5"]["accuracy"] for c in vlm_eval.CORRUPTION_TYPES]))
        method_seed_means.append(np.mean([method[c]["online_accuracy"] for c in vlm_eval.CORRUPTION_TYPES]))
    for corruption in vlm_eval.CORRUPTION_TYPES:
        base_values, method_values = [], []
        for seed in SEEDS:
            baseline = next(iter(json.loads(
                (BASE / f"cub200c_fixed_lambda1.0_seed{seed}.json").read_text()
            )["results"].values()))
            method = json.loads((RESULTS / f"seed{seed}.json").read_text())["results"]
            base_values.append(100 * baseline[corruption]["5"]["accuracy"])
            method_values.append(method[corruption]["online_accuracy"])
        per_corruption[corruption] = {
            "prototta": float(np.mean(base_values)),
            "vlm_guarded": float(np.mean(method_values)),
            "delta": float(np.mean(method_values) - np.mean(base_values)),
        }
    deltas = np.asarray([row["delta"] for row in per_corruption.values()])
    rng = np.random.default_rng(20260902)
    bootstrap = rng.choice(deltas, (200_000, len(deltas)), replace=True).mean(1)
    payload = {
        "schema_version": 1, "method": "confidence-guarded VLM-Purified ProtoTTA",
        "seeds": SEEDS, "prototta_mean": float(np.mean(base_seed_means)),
        "vlm_guarded_mean": float(np.mean(method_seed_means)),
        "vlm_guarded_std": float(np.std(method_seed_means, ddof=1)),
        "delta": float(np.mean(method_seed_means) - np.mean(base_seed_means)),
        "wins": int((deltas > 0).sum()),
        "significance": {
            "unit": "13 matched corruption blocks",
            "bootstrap_95pct_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "exact_sign_flip_p": exact_sign_flip(deltas),
            "wilcoxon_p": float(stats.wilcoxon(deltas).pvalue),
            "paired_t_p": float(stats.ttest_1samp(deltas, 0).pvalue),
        },
        "per_corruption": per_corruption,
    }
    write_json(RESULTS / "summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
