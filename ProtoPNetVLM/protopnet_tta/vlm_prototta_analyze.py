"""Label-revealed final analysis for VLM-guided ProtoTTA."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .vlm_prototta_common import (
    CORRUPTIONS,
    SEEDS,
    atomic_write_json,
    load_json,
    select_prediction,
    sha256_lines,
)


METHODS = (
    "frozen_source", "prototta", "msp_selector", "image_vlm_selector",
    "board_vlm_selector", "oracle",
)


def paired_tests(differences: list[float], bootstrap_seed: int = 0) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (13,):
        raise ValueError("paired tests require 13 corruption-average differences")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(values), size=(200_000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    observed = abs(float(values.mean()))
    sign_flipped_means = np.asarray([
        np.mean(values * np.asarray(signs, dtype=np.float64))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    sign_flip_p = float(np.mean(np.abs(sign_flipped_means) >= observed - 1e-15))
    if np.allclose(values, 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        t_statistic, t_p = 0.0, 1.0
        t_degenerate = True
    else:
        wilcoxon = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        wilcoxon_statistic, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        if float(values.std(ddof=1)) == 0.0:
            t_statistic = None
            t_p = 0.0
            t_degenerate = True
        else:
            paired_t = stats.ttest_1samp(values, popmean=0.0)
            t_statistic, t_p = float(paired_t.statistic), float(paired_t.pvalue)
            t_degenerate = False
    return {
        "n_corruptions": 13,
        "mean_accuracy_difference": float(values.mean()),
        "bootstrap_samples": 200_000,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "exact_two_sided_sign_flip": {
            "enumerated_sign_patterns": 8192,
            "p_value": sign_flip_p,
        },
        "wilcoxon_signed_rank": {
            "statistic": wilcoxon_statistic, "p_value": wilcoxon_p,
        },
        "paired_t_test": {
            "statistic": t_statistic, "p_value": t_p,
            "degenerate_zero_variance": t_degenerate,
        },
    }


def _score_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete VLM scores: {path}")
    return payload["decisions"]


def _score_dir(stream_dir: Path, score_tag: str) -> Path:
    return stream_dir / "vlm_scores" / score_tag if score_tag else stream_dir


def _gate_all_vlm_outputs(output_dir: Path, score_tag: str = "") -> None:
    """Verify every label-free result exists before any target artifact opens."""
    missing = []
    for seed in SEEDS:
        for corruption in CORRUPTIONS:
            stream_dir = output_dir / f"seed_{seed}" / corruption
            directory = _score_dir(stream_dir, score_tag)
            for name in ("public.json", "scores_image.json", "scores_board.json",
                         "vlm.complete.json"):
                path = stream_dir / name if name == "public.json" else directory / name
                if not path.is_file():
                    missing.append(str(path))
            if not missing:
                marker = load_json(directory / "vlm.complete.json")
                if marker.get("status") != "complete" or marker.get("labels_loaded") is not False:
                    raise RuntimeError(f"invalid VLM completion marker: {directory}")
    if missing:
        raise RuntimeError(
            "refusing label-revealed analysis before every VLM output is saved; "
            f"missing {len(missing)} artifacts, first={missing[0]}"
        )


def analyze(output_dir: Path, score_tag: str = "") -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _gate_all_vlm_outputs(output_dir, score_tag)
    # LABEL-REVEALED PHASE STARTS HERE. The scorer never imports this module.
    predictions: dict[str, dict[int, dict[str, list[int]]]] = {
        method: {seed: {} for seed in SEEDS} for method in METHODS
    }
    targets_by_stream: dict[tuple[int, str], list[int]] = {}
    rollback = {
        "image_vlm_selector": {"total": 0, "helpful": 0, "harmful": 0, "neutral": 0},
        "board_vlm_selector": {"total": 0, "helpful": 0, "harmful": 0, "neutral": 0},
    }
    textual_actions = {
        "image": {"ACCEPT": 0, "ROLLBACK": 0},
        "board": {"ACCEPT": 0, "ROLLBACK": 0},
    }
    for seed in SEEDS:
        for corruption in CORRUPTIONS:
            directory = output_dir / f"seed_{seed}" / corruption
            public = load_json(directory / "public.json")
            score_dir = _score_dir(directory, score_tag)
            image_scores = _score_map(score_dir / "scores_image.json")
            board_scores = _score_map(score_dir / "scores_board.json")
            sealed = load_json(directory / "sealed_targets.json")
            records = public["records"]
            ids = [record["sample_id"] for record in records]
            if sealed["ordered_sample_ids"] != ids:
                raise RuntimeError(f"target/sample pairing mismatch: seed={seed} {corruption}")
            targets = [int(value) for value in sealed["targets"]]
            if sealed["target_hash_sha256"] != sha256_lines(targets):
                raise RuntimeError("sealed target hash mismatch")
            targets_by_stream[(seed, corruption)] = targets
            stream_predictions = {method: [] for method in METHODS}
            changed_ids = {record["sample_id"] for record in records if record["changed"]}
            if set(image_scores) != changed_ids or set(board_scores) != changed_ids:
                raise RuntimeError("VLM decisions do not exactly cover disagreements")
            for record, target in zip(records, targets):
                before = int(record["before_prediction"])
                after = int(record["after_prediction"])
                stream_predictions["frozen_source"].append(before)
                stream_predictions["prototta"].append(after)
                stream_predictions["msp_selector"].append(
                    before if record["before_msp"] > record["after_msp"] else after
                )
                if not record["changed"]:
                    image_selected = board_selected = after
                else:
                    image_decision = image_scores[record["sample_id"]]
                    board_decision = board_scores[record["sample_id"]]
                    textual_actions["image"][image_decision["action"]] += 1
                    textual_actions["board"][board_decision["action"]] += 1
                    image_selected = select_prediction(
                        before, after, int(image_decision["adaptation_score"]),
                        mode="image_only",
                    )
                    board_selected = select_prediction(
                        before, after, int(image_decision["adaptation_score"]),
                        int(board_decision["adaptation_score"]), mode="dual_view",
                    )
                stream_predictions["image_vlm_selector"].append(image_selected)
                stream_predictions["board_vlm_selector"].append(board_selected)
                stream_predictions["oracle"].append(
                    before if before == target else after
                )
                for method, selected in (
                    ("image_vlm_selector", image_selected),
                    ("board_vlm_selector", board_selected),
                ):
                    if before != after and selected == before:
                        rollback[method]["total"] += 1
                        if before == target and after != target:
                            rollback[method]["helpful"] += 1
                        elif before != target and after == target:
                            rollback[method]["harmful"] += 1
                        else:
                            rollback[method]["neutral"] += 1
            for method in METHODS:
                predictions[method][seed][corruption] = stream_predictions[method]

    accuracy: dict[str, dict[int, dict[str, float]]] = {
        method: {seed: {} for seed in SEEDS} for method in METHODS
    }
    for method in METHODS:
        for seed in SEEDS:
            for corruption in CORRUPTIONS:
                pred = np.asarray(predictions[method][seed][corruption])
                target = np.asarray(targets_by_stream[(seed, corruption)])
                accuracy[method][seed][corruption] = float(np.mean(pred == target))

    per_seed = {
        method: {
            str(seed): float(np.mean(list(accuracy[method][seed].values())))
            for seed in SEEDS
        } for method in METHODS
    }
    three_seed = {}
    per_corruption = {}
    for method in METHODS:
        seed_values = np.asarray(list(per_seed[method].values()))
        three_seed[method] = {
            "mean": float(seed_values.mean()),
            "sample_std": float(seed_values.std(ddof=1)),
        }
        per_corruption[method] = {}
        for corruption in CORRUPTIONS:
            values = np.asarray([accuracy[method][seed][corruption] for seed in SEEDS])
            per_corruption[method][corruption] = {
                "seed_0": float(values[0]), "seed_2": float(values[1]),
                "seed_3": float(values[2]), "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
            }

    comparisons = (
        ("image_vs_prototta", "image_vlm_selector", "prototta"),
        ("board_vs_prototta", "board_vlm_selector", "prototta"),
        ("board_vs_image", "board_vlm_selector", "image_vlm_selector"),
        ("msp_vs_prototta", "msp_selector", "prototta"),
        ("msp_vs_image", "msp_selector", "image_vlm_selector"),
        ("msp_vs_board", "msp_selector", "board_vlm_selector"),
    )
    comparison_results = {}
    for name, left, right in comparisons:
        differences = [
            per_corruption[left][corruption]["mean"]
            - per_corruption[right][corruption]["mean"]
            for corruption in CORRUPTIONS
        ]
        comparison_results[name] = {
            "left": left, "right": right,
            "mean_improvement": float(np.mean(differences)),
            "wins": sum(value > 1e-15 for value in differences),
            "ties": sum(abs(value) <= 1e-15 for value in differences),
            "losses": sum(value < -1e-15 for value in differences),
            "per_corruption_differences": dict(zip(CORRUPTIONS, differences)),
            "statistical_tests": paired_tests(differences),
        }

    return {
        "schema_version": 1,
        "score_tag": score_tag,
        "analysis_phase": "label-revealed post-hoc; VLM outputs were complete first",
        "accuracy_scale": "fraction",
        "accuracy": {
            method: {str(seed): by_corruption for seed, by_corruption in seeds.items()}
            for method, seeds in accuracy.items()
        },
        "per_seed_mean_over_13_corruptions": per_seed,
        "three_seed_mean_and_sample_std": three_seed,
        "per_corruption_three_seed": per_corruption,
        "comparisons": comparison_results,
        "rollbacks_label_revealed_post_hoc": rollback,
        "vlm_textual_actions_diagnostic_only": textual_actions,
        "oracle_label_revealed_diagnostic_only": three_seed["oracle"],
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Label-free VLM-guided ProtoTTA evaluation", "",
        "VLM decisions are label-free. Oracle and helpful/harmful/neutral rollback "
        "diagnostics below are explicitly label-revealed post-hoc analyses.", "",
        "## Three-seed summary", "",
        "| Method | Mean accuracy (%) | Sample SD across seed means (pp) |",
        "|---|---:|---:|",
    ]
    labels = {
        "frozen_source": "Frozen source", "prototta": "Normal ProtoTTA",
        "msp_selector": "MSP selector", "image_vlm_selector": "Image-only VLM",
        "board_vlm_selector": "Image + reasoning-board VLM",
        "oracle": "Oracle (post-hoc)",
    }
    for method in METHODS:
        row = summary["three_seed_mean_and_sample_std"][method]
        lines.append(f"| {labels[method]} | {_percent(row['mean'])} | {_percent(row['sample_std'])} |")
    for seed in SEEDS:
        lines.extend(("", f"## Seed {seed}: per-corruption accuracy (%)", ""))
        header = "| Corruption | " + " | ".join(labels[m] for m in METHODS) + " |"
        lines.extend((header, "|---|" + "---:|" * len(METHODS)))
        for corruption in CORRUPTIONS:
            values = [summary["accuracy"][m][str(seed)][corruption] for m in METHODS]
            lines.append(
                f"| {corruption} | " + " | ".join(_percent(v) for v in values) + " |"
            )
    lines.extend(("", "## Per-corruption three-seed mean ± sample SD (%)", ""))
    header = "| Corruption | " + " | ".join(labels[m] for m in METHODS) + " |"
    lines.extend((header, "|---|" + "---:|" * len(METHODS)))
    for corruption in CORRUPTIONS:
        cells = []
        for method in METHODS:
            row = summary["per_corruption_three_seed"][method][corruption]
            cells.append(f"{_percent(row['mean'])} ± {_percent(row['sample_std'])}")
        lines.append(f"| {corruption} | " + " | ".join(cells) + " |")
    lines.extend(("", "## Paired comparisons over 13 corruption means", "",
                  "| Comparison | Mean Δ (pp) | W/T/L | Bootstrap 95% CI (pp) | Sign-flip p | Wilcoxon p | Paired t p |",
                  "|---|---:|---:|---:|---:|---:|---:|"))
    for name, result in summary["comparisons"].items():
        tests = result["statistical_tests"]
        ci = tests["bootstrap_95_ci"]
        lines.append(
            f"| {name} | {_percent(result['mean_improvement'])} | "
            f"{result['wins']}/{result['ties']}/{result['losses']} | "
            f"[{_percent(ci[0])}, {_percent(ci[1])}] | "
            f"{tests['exact_two_sided_sign_flip']['p_value']:.6g} | "
            f"{tests['wilcoxon_signed_rank']['p_value']:.6g} | "
            f"{tests['paired_t_test']['p_value']:.6g} |"
        )
    lines.extend(("", "## Rollback diagnostics (label-revealed post-hoc)", "",
                  "| Selector | Rollbacks | Helpful | Harmful | Neutral |",
                  "|---|---:|---:|---:|---:|"))
    for method, row in summary["rollbacks_label_revealed_post_hoc"].items():
        lines.append(
            f"| {labels[method]} | {row['total']} | {row['helpful']} | "
            f"{row['harmful']} | {row['neutral']} |"
        )
    lines.extend(("", "The oracle is a diagnostic upper bound only and is never part of selection."))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-tag", default="")
    args = parser.parse_args(argv)
    summary = analyze(args.output_dir, args.score_tag)
    destination_dir = (args.output_dir / "analyses" / args.score_tag
                       if args.score_tag else args.output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination_dir / "analysis.json", summary)
    report = render_report(summary)
    destination = destination_dir / "report.md"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    print(f"analysis={destination_dir / 'analysis.json'}")
    print(f"report={destination}")


if __name__ == "__main__":
    main()
