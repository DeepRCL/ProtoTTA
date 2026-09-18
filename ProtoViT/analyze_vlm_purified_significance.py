#!/usr/bin/env python3
"""Block-aware paired significance tests for VLM-purified ProtoTTA.

Each corruption is one paired block.  This is deliberately more conservative
than pretending that images in a continuously adapted stream are independent.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from semantic_prototype_guidance import write_json


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results/semantic_prototta/purified_main/summary.json"
OUTPUT = ROOT / "results/semantic_prototta/purified_main/significance.json"


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    """Two-sided exact randomization test of the paired mean difference."""
    observed = abs(float(deltas.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        null.append(abs(float(np.mean(deltas * np.asarray(signs)))))
    null = np.asarray(null)
    return float(np.mean(null >= observed - 1e-12))


def bootstrap_ci(deltas: np.ndarray, seed: int = 20260901) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(200_000, len(deltas)), replace=True).mean(1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    names = list(summary["per_corruption"])
    deltas = np.asarray(
        [summary["per_corruption"][name]["delta"] for name in names],
        dtype=np.float64,
    )
    ttest = stats.ttest_1samp(deltas, 0.0)
    wilcoxon = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
    payload = {
        "schema_version": 1,
        "unit_of_analysis": "13 matched corruption-level mean accuracies; each mean averages three seeds",
        "reason": "corruptions are treated as independent blocks because samples within each continuously adapted stream are dependent",
        "num_blocks": len(deltas),
        "mean_delta_percentage_points": float(deltas.mean()),
        "median_delta_percentage_points": float(np.median(deltas)),
        "wins": int((deltas > 0).sum()),
        "losses": int((deltas < 0).sum()),
        "bootstrap_95pct_ci_mean_delta": bootstrap_ci(deltas),
        "exact_paired_sign_flip_two_sided_p": exact_sign_flip_pvalue(deltas),
        "wilcoxon_signed_rank_two_sided": {
            "statistic": float(wilcoxon.statistic),
            "p": float(wilcoxon.pvalue),
        },
        "paired_t_test_two_sided": {
            "statistic": float(ttest.statistic),
            "p": float(ttest.pvalue),
        },
        "per_corruption_delta": dict(zip(names, map(float, deltas))),
    }
    write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
