"""Label-revealed final analysis for VLM-guided ProtoTTA.

Reports the accuracy of ProtoTTA and of ProtoTTA with the judge, their
difference, and wins/ties/losses over the 13 corruptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .vlm_prototta_common import (
    CORRUPTIONS,
    SEEDS,
    atomic_write_json,
    load_json,
    select_prediction,
    sha256_lines,
)


METHODS = ("prototta", "board_vlm_selector")


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
                stream_predictions["prototta"].append(after)
                if not record["changed"]:
                    board_selected = after
                else:
                    image_decision = image_scores[record["sample_id"]]
                    board_decision = board_scores[record["sample_id"]]
                    board_selected = select_prediction(
                        before, after, int(image_decision["adaptation_score"]),
                        int(board_decision["adaptation_score"]), mode="dual_view",
                    )
                stream_predictions["board_vlm_selector"].append(board_selected)
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
        ("board_vs_prototta", "board_vlm_selector", "prototta"),
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
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def render_report(summary: dict[str, Any]) -> str:
    labels = {"prototta": "ProtoTTA", "board_vlm_selector": "ProtoTTA + Judge"}
    lines = [
        "# Label-free judge intervention", "",
        "| Method | Mean accuracy (%) | Sample SD across seed means (pp) |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        row = summary["three_seed_mean_and_sample_std"][method]
        lines.append(f"| {labels[method]} | {_percent(row['mean'])} | {_percent(row['sample_std'])} |")
    result = summary["comparisons"]["board_vs_prototta"]
    lines.extend((
        "",
        f"Difference (pp): {_percent(result['mean_improvement'])}; "
        f"W/T/L over 13 corruptions: {result['wins']}/{result['ties']}/{result['losses']}",
    ))
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
