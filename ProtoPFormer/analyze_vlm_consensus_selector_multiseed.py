#!/usr/bin/env python3
"""Aggregate the frozen VLM rollback selector over ProtoPFormer seeds 0, 2, and 3."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from noise_utils import CORRUPTION_TYPES


ROOT = Path(__file__).resolve().parent
SEEDS = (0, 2, 3)
IMAGE_THRESHOLD = -4
BOARD_THRESHOLD = -3
SOURCES = {
    0: ROOT / "results/vlm_consensus_selector",
    2: ROOT / "results/vlm_consensus_selector_multiseed/seed2",
    3: ROOT / "results/vlm_consensus_selector_multiseed/seed3",
}
OUTPUT = ROOT / "results/vlm_consensus_selector_multiseed/summary_3seed.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def paired_summary(values: np.ndarray, rng: np.random.Generator) -> dict:
    boots = rng.choice(values, (200_000, len(values)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "wins": int((values > 0).sum()),
        "bootstrap_95pct_ci": [
            float(value) for value in np.quantile(boots, [0.025, 0.975])
        ],
        "exact_sign_flip_p": exact_sign_flip(values),
        "wilcoxon_p": float(stats.wilcoxon(values).pvalue),
        "paired_t_p": float(stats.ttest_1samp(values, 0.0).pvalue),
    }


def main() -> None:
    rows: dict[int, dict[str, dict[str, float]]] = {}
    rollback_counts = {
        "image_only": {str(seed): 0 for seed in SEEDS},
        "with_boards": {str(seed): 0 for seed in SEEDS},
    }
    sample_counts = {str(seed): 0 for seed in SEEDS}

    for seed in SEEDS:
        source = SOURCES[seed]
        rows[seed] = {}
        for corruption in CORRUPTION_TYPES:
            export = read_json(
                source / "export" / f"seed{seed}" / f"{corruption}.json"
            )
            image = read_json(
                source / "scores/image_predictions" / f"{corruption}.json"
            )
            board = read_json(
                source / "scores/full_reasoning" / f"{corruption}.json"
            )
            if not export.get("complete"):
                raise RuntimeError(f"Incomplete export: seed={seed} {corruption}")
            expected = set(export["evidence"])
            if set(image.get("entries", {})) != expected:
                raise RuntimeError(f"Incomplete image scores: seed={seed} {corruption}")
            if set(board.get("entries", {})) != expected:
                raise RuntimeError(f"Incomplete board scores: seed={seed} {corruption}")

            correct = {"prototta": 0, "image_only": 0, "with_boards": 0}
            for key, record in export["records"].items():
                changed = (
                    record["before_prediction_index"]
                    != record["after_prediction_index"]
                )
                image_rejects = changed and (
                    int(image["entries"][key]["adaptation_score"])
                    <= IMAGE_THRESHOLD
                )
                board_confirms = image_rejects and (
                    int(board["entries"][key]["adaptation_score"])
                    <= BOARD_THRESHOLD
                )
                correct["prototta"] += int(record["after_correct"])
                correct["image_only"] += int(
                    record["before_correct"] if image_rejects else record["after_correct"]
                )
                correct["with_boards"] += int(
                    record["before_correct"] if board_confirms else record["after_correct"]
                )
                rollback_counts["image_only"][str(seed)] += int(image_rejects)
                rollback_counts["with_boards"][str(seed)] += int(board_confirms)

            n = len(export["records"])
            sample_counts[str(seed)] += n
            rows[seed][corruption] = {
                name: 100.0 * value / n for name, value in correct.items()
            }

    per_seed = {}
    for seed in SEEDS:
        means = {
            method: float(np.mean([rows[seed][c][method] for c in CORRUPTION_TYPES]))
            for method in ("prototta", "image_only", "with_boards")
        }
        per_seed[str(seed)] = {
            **means,
            "image_delta": means["image_only"] - means["prototta"],
            "board_delta": means["with_boards"] - means["prototta"],
            "boards_over_image": means["with_boards"] - means["image_only"],
        }

    per_corruption = {}
    for corruption in CORRUPTION_TYPES:
        means = {
            method: float(np.mean([rows[s][corruption][method] for s in SEEDS]))
            for method in ("prototta", "image_only", "with_boards")
        }
        per_corruption[corruption] = {
            **means,
            "image_delta": means["image_only"] - means["prototta"],
            "board_delta": means["with_boards"] - means["prototta"],
            "boards_over_image": means["with_boards"] - means["image_only"],
        }

    overall = {
        method: float(np.mean([per_seed[str(s)][method] for s in SEEDS]))
        for method in ("prototta", "image_only", "with_boards")
    }
    std = {
        method: float(np.std([per_seed[str(s)][method] for s in SEEDS], ddof=1))
        for method in ("prototta", "image_only", "with_boards")
    }
    base = np.asarray([per_corruption[c]["prototta"] for c in CORRUPTION_TYPES])
    image = np.asarray([per_corruption[c]["image_only"] for c in CORRUPTION_TYPES])
    board = np.asarray([per_corruption[c]["with_boards"] for c in CORRUPTION_TYPES])
    rng = np.random.default_rng(20260911)

    payload = {
        "schema_version": 1,
        "method": "frozen ProtoViT-developed VLM rollback selector",
        "backbone": "ProtoPFormer",
        "dataset": "Stanford Dogs-C severity 5",
        "seeds": list(SEEDS),
        "ground_truth_visible_to_vlm": False,
        "selection_rule": (
            f"image-only rollback if image score <= {IMAGE_THRESHOLD}; "
            f"dual-view rollback if image score <= {IMAGE_THRESHOLD} and "
            f"reasoning-board score <= {BOARD_THRESHOLD}"
        ),
        "sample_counts": sample_counts,
        "macro_accuracy_mean": overall,
        "macro_accuracy_seed_std": std,
        "per_seed": per_seed,
        "per_corruption": per_corruption,
        "rollback_counts": rollback_counts,
        "significance": {
            "unit": "13 matched corruption means, each averaged across three seeds",
            "image_vs_prototta": paired_summary(image - base, rng),
            "boards_vs_prototta": paired_summary(board - base, rng),
            "boards_vs_image": paired_summary(board - image, rng),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
