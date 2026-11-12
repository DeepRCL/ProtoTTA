#!/usr/bin/env python3
"""Table 8 (ProtoViT/CUB-200-C): accuracy of ProtoTTA with and without the judge.

Inputs are the per-seed exports and reasoning-board scores written by
``fullset_vlm_gate.py --output-dir <seed dir> export --seed <seed>`` and
``fullset_vlm_gate.py --output-dir <seed dir> score --seed <seed> --condition
full_reasoning`` (one output directory per seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import fullset_vlm_gate as gate


ROOT = Path(__file__).resolve().parent
BOARD_THRESHOLD = -5
MIN_MSP_DROP = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 3])
    parser.add_argument("--seed-dirs", type=Path, nargs="+", required=True,
                        help="fullset_vlm_gate.py --output-dir of each seed, in the order of --seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-subset", type=Path,
                        default=ROOT / "results" / "vlm_eval" / "subset.json",
                        help="subset.json written by vlm_eval.py; these images are excluded")
    parser.add_argument("--summary-stem", default="summary_3seed")
    parser.add_argument("--report-stem", default="report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must be distinct")
    seed_dirs = [path.resolve() for path in args.seed_dirs]
    if len(seed_dirs) != len(seeds) or len(set(seed_dirs)) != len(seeds):
        raise SystemExit("--seed-dirs needs one distinct directory per seed")
    seed_dir = dict(zip(seeds, seed_dirs))
    excluded = gate.development_keys(args.development_subset)
    exports = {
        seed: {
            corruption: gate.read_json(gate.export_path(seed_dir[seed], corruption, seed))
            for corruption in gate.CORRUPTIONS
        }
        for seed in seeds
    }
    scores = {
        seed: {
            corruption: gate.read_json(
                gate.score_path(seed_dir[seed], "full_reasoning", corruption)
            )
            for corruption in gate.CORRUPTIONS
        }
        for seed in seeds
    }
    for seed in seeds:
        for corruption in gate.CORRUPTIONS:
            exported = exports[seed][corruption]
            if not exported.get("complete"):
                raise RuntimeError(f"incomplete export: seed={seed} {corruption}")
            if "seed" not in exported or int(exported["seed"]) != seed:
                raise RuntimeError(f"export seed mismatch: seed={seed} {corruption}")
            expected = set(exported["evidence"])
            payload = scores[seed][corruption]
            if int(payload.get("seed", seed)) != seed:
                raise RuntimeError(
                    f"score seed mismatch: seed={seed} full_reasoning/{corruption}"
                )
            if not payload.get("complete") or set(payload["entries"]) != expected:
                raise RuntimeError(
                    f"incomplete scores: seed={seed} full_reasoning/{corruption}"
                )

    methods = ("prototta", "with_boards")
    accuracy = {method: {seed: {} for seed in seeds} for method in methods}
    for seed in seeds:
        for corruption in gate.CORRUPTIONS:
            export = exports[seed][corruption]
            board = scores[seed][corruption]["entries"]
            correct = {method: 0 for method in methods}
            n = 0
            for key, record in export["records"].items():
                if key in excluded:
                    continue
                n += 1
                correct["prototta"] += int(record["after_correct"])
                if key not in export["evidence"]:
                    correct["with_boards"] += int(record["after_correct"])
                    continue
                board_rejects = (
                    int(board[key]["adaptation_score"]) <= BOARD_THRESHOLD
                    and float(record["before_msp"]) - float(record["after_msp"])
                    >= MIN_MSP_DROP
                )
                correct["with_boards"] += int(
                    record["before_correct"] if board_rejects else record["after_correct"]
                )
            for method in methods:
                accuracy[method][seed][corruption] = 100.0 * correct[method] / n

    per_seed = {
        method: {
            str(seed): float(np.mean(list(accuracy[method][seed].values())))
            for seed in seeds
        }
        for method in methods
    }
    seed_summary = {
        method: {
            "mean": float(np.mean(list(per_seed[method].values()))),
            "sample_std": float(np.std(list(per_seed[method].values()), ddof=1)),
        }
        for method in methods
    }
    per_corruption = {
        method: {
            corruption: float(np.mean([
                accuracy[method][seed][corruption] for seed in seeds
            ]))
            for corruption in gate.CORRUPTIONS
        }
        for method in methods
    }
    differences = np.asarray([
        per_corruption["with_boards"][corruption] - per_corruption["prototta"][corruption]
        for corruption in gate.CORRUPTIONS
    ])

    payload = {
        "seeds": list(seeds),
        "prototta": seed_summary["prototta"],
        "prototta_with_judge": seed_summary["with_boards"],
        "difference_percentage_points": float(differences.mean()),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate.write_json(args.output_dir / f"{args.summary_stem}.json", payload)
    lines = [
        "Method | Accuracy (%) | SD (pp)",
        "--- | ---: | ---:",
        f"ProtoTTA | {payload['prototta']['mean']:.3f} | {payload['prototta']['sample_std']:.3f}",
        f"ProtoTTA + Judge | {payload['prototta_with_judge']['mean']:.3f} | "
        f"{payload['prototta_with_judge']['sample_std']:.3f}",
        "",
        f"Difference (pp): {payload['difference_percentage_points']:+.3f}; "
        f"W/T/L: {payload['wins']}/{payload['ties']}/{payload['losses']}",
    ]
    (args.output_dir / f"{args.report_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
