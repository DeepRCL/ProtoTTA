#!/usr/bin/env python3
"""Compare the main VLM-purified method with canonical ProtoTTA."""
import json
from pathlib import Path

import numpy as np

import vlm_eval
from semantic_prototype_guidance import write_json


ROOT = Path(__file__).resolve().parent
evaluation = ROOT / "results/semantic_prototta/purified_main"
baseline_dir = ROOT / "results/fixed_lambda_metrics/10819"
seeds = [0, 1, 2]
baseline_seed_means, method_seed_means = [], []
per_corruption = {}
for corruption in vlm_eval.CORRUPTION_TYPES:
    baseline_values, method_values = [], []
    for seed in seeds:
        baseline = json.loads((baseline_dir / f"cub200c_fixed_lambda1.0_seed{seed}.json").read_text())
        baseline = next(iter(baseline["results"].values()))
        baseline_values.append(100.0 * float(baseline[corruption]["5"]["accuracy"]))
        method = json.loads((evaluation / f"purified_main_seed{seed}.json").read_text())
        method_values.append(float(method["results"][corruption]["online_accuracy"]))
    per_corruption[corruption] = {
        "prototta": float(np.mean(baseline_values)),
        "vlm_purified": float(np.mean(method_values)),
        "delta": float(np.mean(method_values) - np.mean(baseline_values)),
    }
for seed in seeds:
    baseline = json.loads((baseline_dir / f"cub200c_fixed_lambda1.0_seed{seed}.json").read_text())
    baseline = next(iter(baseline["results"].values()))
    baseline_seed_means.append(float(np.mean([
        100.0 * baseline[name]["5"]["accuracy"] for name in vlm_eval.CORRUPTION_TYPES
    ])))
    method = json.loads((evaluation / f"purified_main_seed{seed}.json").read_text())["results"]
    method_seed_means.append(float(np.mean([
        method[name]["online_accuracy"] for name in vlm_eval.CORRUPTION_TYPES
    ])))
payload = {
    "schema_version": 1,
    "method": "VLM-Purified ProtoTTA (forced ranking + semantic logits + active semantic loss)",
    "seeds": seeds,
    "prototta_mean": float(np.mean(baseline_seed_means)),
    "prototta_std": float(np.std(baseline_seed_means, ddof=1)),
    "vlm_purified_mean": float(np.mean(method_seed_means)),
    "vlm_purified_std": float(np.std(method_seed_means, ddof=1)),
    "delta": float(np.mean(method_seed_means) - np.mean(baseline_seed_means)),
    "corruption_wins": int(sum(item["delta"] > 0 for item in per_corruption.values())),
    "per_seed": method_seed_means,
    "per_corruption": per_corruption,
}
write_json(evaluation / "summary.json", payload)
print(json.dumps(payload, indent=2))
