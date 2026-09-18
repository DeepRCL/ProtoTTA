#!/usr/bin/env python3
"""Evaluate the frozen ProtoViT dual-view selector on ProtoPFormer."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from noise_utils import CORRUPTION_TYPES


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/vlm_consensus_selector"
IMAGE_THRESHOLD = -4
BOARD_THRESHOLD = -3


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def main() -> None:
    per_corruption = {}
    rollbacks = {"image_only": 0, "with_boards": 0}
    for corruption in CORRUPTION_TYPES:
        export = json.loads(
            (RESULTS / "export/seed0" / f"{corruption}.json").read_text()
        )
        image = json.loads(
            (RESULTS / "scores/image_predictions" / f"{corruption}.json").read_text()
        )["entries"]
        board = json.loads(
            (RESULTS / "scores/full_reasoning" / f"{corruption}.json").read_text()
        )["entries"]
        counts = {"prototta": 0, "image_only": 0, "with_boards": 0, "n": 0}
        for key, record in export["records"].items():
            changed = record["before_prediction_index"] != record["after_prediction_index"]
            image_rejects = changed and int(image[key]["adaptation_score"]) <= IMAGE_THRESHOLD
            board_confirms = image_rejects and int(board[key]["adaptation_score"]) <= BOARD_THRESHOLD
            counts["prototta"] += int(record["after_correct"])
            counts["image_only"] += int(
                record["before_correct"] if image_rejects else record["after_correct"]
            )
            counts["with_boards"] += int(
                record["before_correct"] if board_confirms else record["after_correct"]
            )
            counts["n"] += 1
            rollbacks["image_only"] += int(image_rejects)
            rollbacks["with_boards"] += int(board_confirms)
        acc = {name: 100.0 * counts[name] / counts["n"]
               for name in ("prototta", "image_only", "with_boards")}
        per_corruption[corruption] = {
            **acc,
            "image_delta": acc["image_only"] - acc["prototta"],
            "board_delta": acc["with_boards"] - acc["prototta"],
            "boards_over_image": acc["with_boards"] - acc["image_only"],
        }
    base = np.asarray([row["prototta"] for row in per_corruption.values()])
    image = np.asarray([row["image_only"] for row in per_corruption.values()])
    board = np.asarray([row["with_boards"] for row in per_corruption.values()])
    delta = board - base
    rng = np.random.default_rng(20260903)
    boots = rng.choice(delta, (200_000, len(delta)), replace=True).mean(axis=1)
    payload = {
        "schema_version": 1,
        "method": "frozen ProtoViT-developed dual-view VLM selector",
        "backbone": "ProtoPFormer",
        "seed": 0,
        "ground_truth_visible_to_vlm": False,
        "selection_rule": (
            f"rollback only if image score <= {IMAGE_THRESHOLD} and "
            f"reasoning-board score <= {BOARD_THRESHOLD}"
        ),
        "macro_accuracy": {
            "prototta": float(base.mean()),
            "image_only_selector": float(image.mean()),
            "dual_view_board_selector": float(board.mean()),
        },
        "delta_vs_prototta": float(delta.mean()),
        "boards_over_image_only": float((board - image).mean()),
        "wins": int((delta > 0).sum()),
        "rollback_counts": rollbacks,
        "per_corruption": per_corruption,
        "significance": {
            "unit": "13 matched corruption blocks",
            "bootstrap_95pct_ci": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
            "exact_sign_flip_p": exact_sign_flip(delta),
            "wilcoxon_p": float(stats.wilcoxon(delta).pvalue),
            "paired_t_p": float(stats.ttest_1samp(delta, 0.0).pvalue),
        },
    }
    output = RESULTS / "summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
