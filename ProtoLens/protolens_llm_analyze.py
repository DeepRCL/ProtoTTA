#!/usr/bin/env python3
"""Analyze the label-free ProtoLens LLM supervisor over corruption settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

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
    "frozen_source",
    "prototta",
    "msp_selector",
    "llm_output_selector",
    "llm_prototype_selector",
    "oracle",
)


def exact_sign_flip(values: np.ndarray) -> float:
    """Return an exact two-sided sign-flip p-value."""
    observed = abs(float(values.mean()))
    count = 0
    extreme = 0
    width = len(values)
    bits = np.arange(width, dtype=np.uint64)
    for start in range(0, 1 << width, 65536):
        numbers = np.arange(
            start, min(start + 65536, 1 << width), dtype=np.uint64
        )
        signs = 2.0 * ((numbers[:, None] >> bits) & 1).astype(np.float64) - 1.0
        means = np.mean(signs * values[None, :], axis=1)
        extreme += int(np.sum(np.abs(means) >= observed - 1e-15))
        count += len(numbers)
    return extreme / count


def paired_tests(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if np.allclose(array, 0):
        wilcoxon_p = t_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(array, zero_method="wilcox").pvalue)
        t_p = float(stats.ttest_1samp(array, popmean=0.0).pvalue)
    return {
        "n_settings": len(array),
        "mean_delta": float(array.mean()),
        "exact_sign_flip_p": exact_sign_flip(array),
        "wilcoxon_p": wilcoxon_p,
        "paired_t_p": t_p,
    }


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
    rollbacks = {
        "llm_output_selector": {
            "total": 0, "helpful": 0, "harmful": 0, "neutral": 0
        },
        "llm_prototype_selector": {
            "total": 0, "helpful": 0, "harmful": 0, "neutral": 0
        },
    }
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
                output_selected = prototype_selected = after
            else:
                output_score = int(
                    output_scores[record["sample_id"]]["adaptation_score"]
                )
                prototype_score = int(
                    prototype_scores[record["sample_id"]]["adaptation_score"]
                )
                output_selected = select_prediction(
                    before, after, output_score, mode="output_only"
                )
                prototype_selected = select_prediction(
                    before,
                    after,
                    output_score,
                    prototype_score,
                    mode="dual_view",
                )
            selected = {
                "frozen_source": before,
                "prototta": after,
                "msp_selector": (
                    before
                    if float(record["before_msp"]) > float(record["after_msp"])
                    else after
                ),
                "llm_output_selector": output_selected,
                "llm_prototype_selector": prototype_selected,
                "oracle": before if before == target else after,
            }
            for method, prediction in selected.items():
                correct[method] += int(prediction == target)
            for method in ("llm_output_selector", "llm_prototype_selector"):
                if before != after and selected[method] == before:
                    row = rollbacks[method]
                    row["total"] += 1
                    if before == target and after != target:
                        row["helpful"] += 1
                    elif before != target and after == target:
                        row["harmful"] += 1
                    else:
                        row["neutral"] += 1

        setting = f"{corruption}_s{severity}"
        for method in METHODS:
            accuracy[method][setting] = correct[method] / len(records)
        total_samples += len(records)
        total_disagreements += len(changed_ids)

    mean_accuracy = {
        method: float(np.mean(list(values.values())))
        for method, values in accuracy.items()
    }
    comparisons = {}
    for name, left, right in (
        ("output_vs_prototta", "llm_output_selector", "prototta"),
        ("prototype_vs_prototta", "llm_prototype_selector", "prototta"),
        ("prototype_vs_output", "llm_prototype_selector", "llm_output_selector"),
        ("msp_vs_prototta", "msp_selector", "prototta"),
    ):
        differences = [
            accuracy[left][setting] - accuracy[right][setting]
            for setting in accuracy[left]
        ]
        comparisons[name] = {
            "left": left,
            "right": right,
            "wins": sum(value > 1e-15 for value in differences),
            "ties": sum(abs(value) <= 1e-15 for value in differences),
            "losses": sum(value < -1e-15 for value in differences),
            "per_setting_differences": dict(zip(accuracy[left], differences)),
            "statistical_tests": paired_tests(differences),
        }
    return {
        "schema_version": 1,
        "analysis_phase": "labels opened after all LLM outputs completed",
        "llm_input": "text only; labels and correctness withheld",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "sample_counts": {
            "total": total_samples,
            "disagreements": total_disagreements,
        },
        "accuracy": accuracy,
        "mean_accuracy": mean_accuracy,
        "comparisons": comparisons,
        "rollbacks_label_revealed_post_hoc": rollbacks,
        "oracle_label_revealed_diagnostic_only": mean_accuracy["oracle"],
    }


def render(summary: dict[str, Any]) -> str:
    labels = {
        "frozen_source": "Frozen source",
        "prototta": "ProtoTTA",
        "msp_selector": "MSP selector",
        "llm_output_selector": "Text-only LLM",
        "llm_prototype_selector": "Text + prototype-evidence LLM",
        "oracle": "Oracle (post-hoc)",
    }
    lines = [
        "# Label-free LLM-guided ProtoLens",
        "",
        "Ground truth was sealed until all LLM decisions were complete.",
        "",
        "Method | Mean accuracy (%)",
        "--- | ---:",
    ]
    for method in METHODS:
        lines.append(f"{labels[method]} | {100 * summary['mean_accuracy'][method]:.3f}")
    lines.extend([
        "",
        "Comparison | Delta (pp) | W/T/L | sign-flip p",
        "--- | ---: | ---: | ---:",
    ])
    for name, result in summary["comparisons"].items():
        tests = result["statistical_tests"]
        lines.append(
            f"{name} | {100 * tests['mean_delta']:.3f} | "
            f"{result['wins']}/{result['ties']}/{result['losses']} | "
            f"{tests['exact_sign_flip_p']:.6g}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    summary = analyze(output_dir)
    atomic_write_json(output_dir / "analysis.json", summary)
    temporary = output_dir / ".report.md.tmp"
    temporary.write_text(render(summary), encoding="utf-8")
    temporary.replace(output_dir / "report.md")
    print(json.dumps(summary["mean_accuracy"], indent=2))


if __name__ == "__main__":
    main()
