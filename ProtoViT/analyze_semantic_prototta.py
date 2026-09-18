#!/usr/bin/env python3
"""Aggregate the three-seed semantic ProtoTTA experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_semantic_prototta import CORRUPTIONS, VARIANTS
from semantic_prototype_guidance import write_json


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "results/semantic_prototta/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/semantic_prototta/summary.json")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results/fixed_lambda_metrics/10819")
    args = parser.parse_args()
    baseline_payloads = [json.loads((args.baseline_dir / f"cub200c_fixed_lambda1.0_seed{seed}.json").read_text()) for seed in (0, 1, 2)]
    baseline_by_seed = []
    baseline_by_corruption = {name: [] for name in CORRUPTIONS}
    for payload in baseline_payloads:
        mode_results = next(iter(payload["results"].values()))
        values = []
        for corruption in CORRUPTIONS:
            accuracy = 100.0 * float(mode_results[corruption]["5"]["accuracy"])
            values.append(accuracy)
            baseline_by_corruption[corruption].append(accuracy)
        baseline_by_seed.append(float(np.mean(values)))
    baseline_corruption_mean = {name: float(np.mean(values)) for name, values in baseline_by_corruption.items()}
    summary = {
        "schema_version": 1,
        "seeds": [0, 1, 2],
        "baseline": {
            "name": "canonical ProtoTTA (fixed lambda=1.0/pure prototype objective)",
            "mean_accuracy": float(np.mean(baseline_by_seed)),
            "std_across_seeds": float(np.std(baseline_by_seed, ddof=1)),
            "per_seed_accuracy": baseline_by_seed,
            "per_corruption": baseline_corruption_mean,
        },
        "variants": {},
    }
    for variant in VARIANTS:
        payloads = [json.loads((args.evaluation_dir / f"{variant}_seed{seed}.json").read_text()) for seed in (0, 1, 2)]
        per_seed = []
        per_corruption = {}
        for corruption in CORRUPTIONS:
            values = [payload["results"][corruption]["online_accuracy"] for payload in payloads]
            post = [payload["results"][corruption]["immediate_post_update_accuracy"] for payload in payloads]
            per_corruption[corruption] = {
                "online_mean": float(np.mean(values)), "online_std": float(np.std(values, ddof=1)),
                "post_mean": float(np.mean(post)), "post_std": float(np.std(post, ddof=1)),
            }
        for payload in payloads:
            per_seed.append(float(np.mean([payload["results"][name]["online_accuracy"] for name in CORRUPTIONS])))
        summary["variants"][variant] = {
            "mean_accuracy": float(np.mean(per_seed)),
            "std_across_seeds": float(np.std(per_seed, ddof=1)),
            "per_seed_accuracy": per_seed,
            "delta_vs_prototta": float(np.mean(per_seed) - np.mean(baseline_by_seed)),
            "corruption_wins": int(sum(
                per_corruption[name]["online_mean"] > baseline_corruption_mean[name]
                for name in CORRUPTIONS
            )),
            "per_corruption": per_corruption,
            "mean_rollback_rate": float(np.mean([
                payload["results"][name]["rollback_rate"] for payload in payloads for name in CORRUPTIONS
            ])),
        }
    write_json(args.output, summary)
    print(f"{'ProtoTTA baseline':24s} {summary['baseline']['mean_accuracy']:.3f} +/- {summary['baseline']['std_across_seeds']:.3f}")
    for variant, values in summary["variants"].items():
        print(f"{variant:24s} {values['mean_accuracy']:.3f} +/- {values['std_across_seeds']:.3f} "
              f"delta={values['delta_vs_prototta']:+.3f} wins={values['corruption_wins']}/13")


if __name__ == "__main__":
    main()
