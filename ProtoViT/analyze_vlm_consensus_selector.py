#!/usr/bin/env python3
"""Evaluate a conservative dual-view VLM selector on held-out CUB-200-C.

The adapted prediction is retained by default.  A frozen-source prediction is
used only when both independent VLM audits strongly reject adaptation:

* image/prediction audit adaptation score <= -4;
* reasoning-board audit adaptation score <= -3.

Ground truth is read only here, after all VLM decisions have been saved.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

import fullset_vlm_gate as gate


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/vlm_fullset_gate"
OUTPUT = RESULTS / "consensus_selector_metrics.json"
IMAGE_THRESHOLD = -4
BOARD_THRESHOLD = -3


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def main() -> None:
    excluded = gate.development_keys(ROOT / "results/vlm_eval/subset.json")
    per_corruption = {}
    baseline_correct = image_correct = board_correct = total = 0
    rollback_counts = {"image_only": 0, "with_boards": 0}
    for corruption in gate.CORRUPTIONS:
        export = gate.read_json(gate.export_path(RESULTS, corruption))
        image_scores = gate.read_json(
            gate.score_path(RESULTS, "image_predictions", corruption)
        )["entries"]
        board_scores = gate.read_json(
            gate.score_path(RESULTS, "full_reasoning", corruption)
        )["entries"]
        counts = {"baseline": 0, "image_only": 0, "with_boards": 0, "n": 0}
        for key, record in export["records"].items():
            if key in excluded:
                continue
            changed = record["before_prediction_index"] != record["after_prediction_index"]
            image_rejects = changed and (
                int(image_scores[key]["adaptation_score"]) <= IMAGE_THRESHOLD
            )
            board_confirms = image_rejects and (
                int(board_scores[key]["adaptation_score"]) <= BOARD_THRESHOLD
            )
            counts["baseline"] += int(record["after_correct"])
            counts["image_only"] += int(
                record["before_correct"] if image_rejects else record["after_correct"]
            )
            counts["with_boards"] += int(
                record["before_correct"] if board_confirms else record["after_correct"]
            )
            counts["n"] += 1
            rollback_counts["image_only"] += int(image_rejects)
            rollback_counts["with_boards"] += int(board_confirms)
        values = {
            name: 100.0 * counts[name] / counts["n"]
            for name in ("baseline", "image_only", "with_boards")
        }
        per_corruption[corruption] = {
            **values,
            "image_delta": values["image_only"] - values["baseline"],
            "board_delta": values["with_boards"] - values["baseline"],
            "boards_over_image": values["with_boards"] - values["image_only"],
        }
        baseline_correct += counts["baseline"]
        image_correct += counts["image_only"]
        board_correct += counts["with_boards"]
        total += counts["n"]

    baseline = np.asarray([x["baseline"] for x in per_corruption.values()])
    image = np.asarray([x["image_only"] for x in per_corruption.values()])
    board = np.asarray([x["with_boards"] for x in per_corruption.values()])
    delta = board - baseline
    rng = np.random.default_rng(20260903)
    boots = rng.choice(delta, (200_000, len(delta)), replace=True).mean(axis=1)
    payload = {
        "schema_version": 1,
        "method": "conservative dual-view VLM selector",
        "vlm_model_id": gate.MODEL_ID,
        "ground_truth_visible_to_vlm": False,
        "selection_rule": {
            "default": "retain continuous ProtoTTA prediction",
            "rollback_to_frozen_source_if": (
                f"image adaptation score <= {IMAGE_THRESHOLD} AND "
                f"reasoning-board adaptation score <= {BOARD_THRESHOLD}"
            ),
        },
        "split": "full CUB-200-C severity-5 test set excluding 100 prompt-development samples",
        "n": total,
        "macro_accuracy": {
            "prototta": float(baseline.mean()),
            "image_only_selector": float(image.mean()),
            "dual_view_board_selector": float(board.mean()),
        },
        "micro_correct": {
            "prototta": baseline_correct,
            "image_only_selector": image_correct,
            "dual_view_board_selector": board_correct,
        },
        "delta_vs_prototta": float(delta.mean()),
        "boards_over_image_only": float((board - image).mean()),
        "corruption_wins_vs_prototta": int((delta > 0).sum()),
        "rollback_counts": rollback_counts,
        "per_corruption": per_corruption,
        "significance": {
            "unit": "13 matched corruption blocks",
            "bootstrap_95pct_ci": [
                float(value) for value in np.quantile(boots, [0.025, 0.975])
            ],
            "exact_sign_flip_p": exact_sign_flip(delta),
            "wilcoxon_p": float(stats.wilcoxon(delta).pvalue),
            "paired_t_p": float(stats.ttest_1samp(delta, 0.0).pvalue),
        },
        "development_note": (
            "Thresholds are a ProtoViT development choice; untouched-backbone "
            "transfer is required before claiming generalization."
        ),
    }
    gate.write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
