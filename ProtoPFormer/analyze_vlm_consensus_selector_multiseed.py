#!/usr/bin/env python3
"""Aggregate the frozen VLM rollback selector over ProtoPFormer seeds 0, 2, and 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(
            f"--seed{seed}-dir", type=Path, default=SOURCES[seed],
            help=f"--output-dir used for the seed-{seed} export and judge scores",
        )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {seed: getattr(args, f"seed{seed}_dir") for seed in SEEDS}
    output = args.output
    rows: dict[int, dict[str, dict[str, float]]] = {}
    sample_counts = {str(seed): 0 for seed in SEEDS}

    for seed in SEEDS:
        source = sources[seed]
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

            correct = {"prototta": 0, "with_boards": 0}
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
                correct["with_boards"] += int(
                    record["before_correct"] if board_confirms else record["after_correct"]
                )

            n = len(export["records"])
            sample_counts[str(seed)] += n
            rows[seed][corruption] = {
                name: 100.0 * value / n for name, value in correct.items()
            }

    per_seed = {}
    for seed in SEEDS:
        means = {
            method: float(np.mean([rows[seed][c][method] for c in CORRUPTION_TYPES]))
            for method in ("prototta", "with_boards")
        }
        per_seed[str(seed)] = {
            **means,
            "board_delta": means["with_boards"] - means["prototta"],
        }

    per_corruption = {}
    for corruption in CORRUPTION_TYPES:
        means = {
            method: float(np.mean([rows[s][corruption][method] for s in SEEDS]))
            for method in ("prototta", "with_boards")
        }
        per_corruption[corruption] = {
            **means,
            "board_delta": means["with_boards"] - means["prototta"],
        }

    overall = {
        method: float(np.mean([per_seed[str(s)][method] for s in SEEDS]))
        for method in ("prototta", "with_boards")
    }
    std = {
        method: float(np.std([per_seed[str(s)][method] for s in SEEDS], ddof=1))
        for method in ("prototta", "with_boards")
    }
    deltas = [per_corruption[c]["board_delta"] for c in CORRUPTION_TYPES]

    payload = {
        "schema_version": 1,
        "method": "frozen ProtoViT-developed VLM rollback selector",
        "backbone": "ProtoPFormer",
        "dataset": "Stanford Dogs-C severity 5",
        "seeds": list(SEEDS),
        "ground_truth_visible_to_vlm": False,
        "selection_rule": (
            f"rollback if image score <= {IMAGE_THRESHOLD} and "
            f"reasoning-board score <= {BOARD_THRESHOLD}"
        ),
        "sample_counts": sample_counts,
        "macro_accuracy_mean": overall,
        "macro_accuracy_seed_std": std,
        "macro_accuracy_delta": overall["with_boards"] - overall["prototta"],
        "wins_ties_losses": [
            sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)
        ],
        "per_seed": per_seed,
        "per_corruption": per_corruption,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
