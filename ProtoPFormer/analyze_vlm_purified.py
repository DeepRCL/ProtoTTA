#!/usr/bin/env python3
"""Aggregate ProtoPFormer VLM purification against the strongest baseline."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from noise_utils import CORRUPTION_TYPES

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "paper_sweep_results/main_protocol_lambda_sweep/11953"
RESULTS = ROOT / "results/vlm_purified/evaluation"
SEEDS = [0, 2, 3]


def exact_sign_flip(deltas):
    observed = abs(float(np.mean(deltas)))
    null = [abs(float(np.mean(np.asarray(signs) * deltas)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(deltas))]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def main():
    per_corruption = {}
    base_seed_means, method_seed_means = [], []
    for seed in SEEDS:
        base = json.loads((BASE / f"dogs_c_lambda0.2_seed{seed}.json").read_text())["results"]["proto_tta"]
        method = json.loads((RESULTS / f"seed{seed}.json").read_text())["results"]
        base_seed_means.append(np.mean([100 * base[c]["5"]["accuracy"] for c in CORRUPTION_TYPES]))
        method_seed_means.append(np.mean([method[c]["online_accuracy"] for c in CORRUPTION_TYPES]))
    for corruption in CORRUPTION_TYPES:
        baseline, method = [], []
        for seed in SEEDS:
            base = json.loads((BASE / f"dogs_c_lambda0.2_seed{seed}.json").read_text())["results"]["proto_tta"]
            result = json.loads((RESULTS / f"seed{seed}.json").read_text())["results"]
            baseline.append(100 * base[corruption]["5"]["accuracy"])
            method.append(result[corruption]["online_accuracy"])
        per_corruption[corruption] = {
            "baseline": float(np.mean(baseline)), "vlm_purified": float(np.mean(method)),
            "delta": float(np.mean(method) - np.mean(baseline)),
        }
    deltas = np.asarray([x["delta"] for x in per_corruption.values()])
    rng = np.random.default_rng(20260901)
    boots = rng.choice(deltas, (200_000, len(deltas)), replace=True).mean(1)
    payload = {
        "schema_version": 1, "baseline": "strongest fixed-lambda ProtoPFormer, lambda=0.2",
        "seeds": SEEDS, "baseline_mean": float(np.mean(base_seed_means)),
        "vlm_purified_mean": float(np.mean(method_seed_means)),
        "delta": float(np.mean(method_seed_means) - np.mean(base_seed_means)),
        "wins": int((deltas > 0).sum()), "per_corruption": per_corruption,
        "significance": {
            "unit": "13 matched corruption blocks",
            "bootstrap_95pct_ci": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
            "exact_sign_flip_p": exact_sign_flip(deltas),
            "wilcoxon_p": float(stats.wilcoxon(deltas).pvalue),
            "paired_t_p": float(stats.ttest_1samp(deltas, 0).pvalue),
        },
    }
    output = RESULTS / "summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
