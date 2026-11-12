#!/usr/bin/env python3
"""Analyze the label-free ProtoLens judge intervention over corruption settings.

Each ``--output-dir`` holds the exported and scored streams of one seed
(``protolens_llm_export.py`` and ``protolens_llm_score.py``). For every seed,
accuracy is computed per corruption-severity setting for ProtoTTA and for
ProtoTTA + judge. The summary reports the mean over settings of the seed-mean
accuracies, the sample standard deviation of the seed-level macro-averages, and
the per-setting wins/ties/losses of the judge against ProtoTTA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from protolens_llm_common import (
    CORRUPTIONS,
    SEVERITIES,
    atomic_write_json,
    load_json,
    select_prediction,
    sha256_lines,
    stream_dir,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "llm_prototta"
METHODS = (
    "prototta",
    "llm_prototype_selector",
)


def score_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete scores: {path}")
    return payload["decisions"]


def analyze(output_dir: Path) -> dict[str, Any]:
    streams = []
    missing = []
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            directory = stream_dir(output_dir, corruption, severity)
            for name in (
                "public.json",
                "scores_output.json",
                "scores_prototype.json",
                "llm.complete.json",
            ):
                if not (directory / name).is_file():
                    missing.append(str(directory / name))
            streams.append((corruption, severity, directory))
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} label-free artifacts; first={missing[0]}"
        )
    for _, _, directory in streams:
        marker = load_json(directory / "llm.complete.json")
        if marker.get("status") != "complete" or marker.get("labels_loaded") is not False:
            raise RuntimeError(f"invalid LLM completion marker: {directory}")

    accuracy = {method: {} for method in METHODS}
    total_samples = total_disagreements = 0
    for corruption, severity, directory in streams:
        public = load_json(directory / "public.json")
        output_scores = score_map(directory / "scores_output.json")
        prototype_scores = score_map(directory / "scores_prototype.json")
        records = public["records"]
        changed_ids = {record["sample_id"] for record in records if record["changed"]}
        if set(output_scores) != changed_ids or set(prototype_scores) != changed_ids:
            raise RuntimeError(f"score coverage mismatch: {directory}")

        # Labels are opened only after every label-free artifact is complete.
        sealed = load_json(directory / "sealed_targets.json")
        identifiers = [record["sample_id"] for record in records]
        if sealed["ordered_sample_ids"] != identifiers:
            raise RuntimeError(f"target order mismatch: {directory}")
        targets = [int(value) for value in sealed["targets"]]
        if sealed["target_hash_sha256"] != sha256_lines(targets):
            raise RuntimeError(f"target hash mismatch: {directory}")

        correct = {method: 0 for method in METHODS}
        for record, target in zip(records, targets):
            before = int(record["before_prediction_index"])
            after = int(record["after_prediction_index"])
            if not record["changed"]:
                prototype_selected = after
            else:
                output_score = int(
                    output_scores[record["sample_id"]]["adaptation_score"]
                )
                prototype_score = int(
                    prototype_scores[record["sample_id"]]["adaptation_score"]
                )
                prototype_selected = select_prediction(
                    before,
                    after,
                    output_score,
                    prototype_score,
                    mode="dual_view",
                )
            selected = {
                "prototta": after,
                "llm_prototype_selector": prototype_selected,
            }
            for method, prediction in selected.items():
                correct[method] += int(prediction == target)

        setting = f"{corruption}_s{severity}"
        for method in METHODS:
            accuracy[method][setting] = correct[method] / len(records)
        total_samples += len(records)
        total_disagreements += len(changed_ids)

    mean_accuracy = {
        method: float(np.mean(list(values.values())))
        for method, values in accuracy.items()
    }
    return {
        "sample_counts": {
            "total": total_samples,
            "disagreements": total_disagreements,
        },
        "accuracy": accuracy,
        "mean_accuracy": mean_accuracy,
    }


def summarize(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seeds = list(per_seed)
    settings = list(per_seed[seeds[0]]["accuracy"]["prototta"])
    per_setting_mean = {
        method: {
            setting: float(np.mean([
                per_seed[seed]["accuracy"][method][setting] for seed in seeds
            ]))
            for setting in settings
        }
        for method in METHODS
    }
    seed_means = {
        method: {seed: per_seed[seed]["mean_accuracy"][method] for seed in seeds}
        for method in METHODS
    }
    mean_and_sd = {}
    for method in METHODS:
        values = np.asarray(list(seed_means[method].values()), dtype=np.float64)
        mean_and_sd[method] = {
            "mean": float(np.mean(list(per_setting_mean[method].values()))),
            "sample_std_of_seed_means": (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
        }
    differences = [
        per_setting_mean["llm_prototype_selector"][setting]
        - per_setting_mean["prototta"][setting]
        for setting in settings
    ]
    comparison = {
        "left": "llm_prototype_selector",
        "right": "prototta",
        "wins": sum(value > 1e-15 for value in differences),
        "ties": sum(abs(value) <= 1e-15 for value in differences),
        "losses": sum(value < -1e-15 for value in differences),
        "mean_delta": float(np.mean(differences)),
        "per_setting_differences": dict(zip(settings, differences)),
    }
    return {
        "schema_version": 2,
        "analysis_phase": "labels opened after all LLM outputs completed",
        "llm_input": "text only; labels and correctness withheld",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "runs": seeds,
        "sample_counts": {seed: per_seed[seed]["sample_counts"] for seed in seeds},
        "per_seed_accuracy": {seed: per_seed[seed]["accuracy"] for seed in seeds},
        "per_seed_mean": seed_means,
        "per_setting_mean_over_seeds": per_setting_mean,
        "mean_and_seed_sd": mean_and_sd,
        "judge_vs_prototta": comparison,
    }


def render(summary: dict[str, Any]) -> str:
    labels = {
        "prototta": "ProtoTTA",
        "llm_prototype_selector": "ProtoTTA + Judge",
    }
    lines = [
        "# Label-free judge intervention on ProtoLens / Amazon-C",
        "",
        f"Runs: {', '.join(summary['runs'])}",
        "",
        "Method | Mean accuracy (%) | SD of seed means (pp)",
        "--- | ---: | ---:",
    ]
    for method in METHODS:
        row = summary["mean_and_seed_sd"][method]
        lines.append(
            f"{labels[method]} | {100 * row['mean']:.3f} | "
            f"{100 * row['sample_std_of_seed_means']:.3f}"
        )
    result = summary["judge_vs_prototta"]
    lines.extend([
        "",
        "Comparison | Delta (pp) | W/T/L",
        "--- | ---: | ---:",
        f"Judge vs ProtoTTA | {100 * result['mean_delta']:+.3f} | "
        f"{result['wins']}/{result['ties']}/{result['losses']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, nargs="+", default=[DEFAULT_OUTPUT],
        help="One directory per seed, as passed to the export and score scripts",
    )
    parser.add_argument(
        "--summary-dir", type=Path, default=None,
        help="Where to write analysis.json and report.md "
             "(default: the output directory when only one is given)",
    )
    args = parser.parse_args()
    output_dirs = [path.resolve() for path in args.output_dir]
    if args.summary_dir is None:
        if len(output_dirs) != 1:
            parser.error("--summary-dir is required with several --output-dir values")
        summary_dir = output_dirs[0]
    else:
        summary_dir = args.summary_dir.resolve()
    per_seed = {str(path): analyze(path) for path in output_dirs}
    summary = summarize(per_seed)
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_dir / "analysis.json", summary)
    temporary = summary_dir / ".report.md.tmp"
    temporary.write_text(render(summary), encoding="utf-8")
    temporary.replace(summary_dir / "report.md")
    print(render(summary))


if __name__ == "__main__":
    main()
