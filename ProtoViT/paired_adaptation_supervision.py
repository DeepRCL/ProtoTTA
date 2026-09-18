#!/usr/bin/env python3
"""Paired, label-blind VLM supervision of test-time adaptation.

The VLM compares anonymized classifier states A and B.  It sees both predicted
labels, while the ground-truth label and before/after identity remain hidden.
Three input conditions isolate the value of prototype reasoning boards:

* predictions_only: the two predicted class names
* image_predictions: corrupted image plus the two predicted class names
* full_reasoning: image, labels, and predicted/any-class boards for both states

Scoring uses the next-token preference probability for A versus B, avoiding
long free-form generation and JSON parsing failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from failure_detection import (
    ANY_CLASS_BOARD_NAME,
    PREDICTED_BOARD_NAME,
    sanitize_board,
)


METHOD_ORDER = ["tent", "eata", "prototta"]
CONDITION_ORDER = ["predictions_only", "image_predictions", "full_reasoning"]
DISPLAY_NAMES = {"tent": "Tent", "eata": "EATA", "prototta": "ProtoTTA"}
CONDITION_NAMES = {
    "predictions_only": "Predictions only",
    "image_predictions": "Image + predictions",
    "full_reasoning": "Image + predictions + reasoning boards",
}
LOGGER = logging.getLogger("paired_adaptation_supervision")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=script_dir / "results" / "vlm_eval"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "vlm_eval" / "paired_supervision",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--method", choices=METHOD_ORDER, required=True)
    score.add_argument("--condition", choices=CONDITION_ORDER, required=True)
    score.add_argument("--vlm-model-id", default="Qwen/Qwen3-VL-32B-Thinking")
    score.add_argument("--max-samples", type=int, default=None)
    score.add_argument("--overwrite", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--bootstrap-resamples", type=int, default=5000)
    analyze.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def sample_meta_map(results_dir: Path, method: str) -> Dict[str, Dict]:
    root = results_dir / method / "samples"
    output = {}
    if not root.exists():
        return output
    for meta_path in sorted(root.glob("*/03_meta.json")):
        output[meta_path.parent.name] = read_json(meta_path)
    return output


def changed_pairs(results_dir: Path, method: str) -> List[Dict]:
    before = sample_meta_map(results_dir, "unadapted")
    after = sample_meta_map(results_dir, method)
    pairs = []
    for sample_id in sorted(set(before) & set(after)):
        before_meta = before[sample_id]
        after_meta = after[sample_id]
        if int(before_meta["predicted_index"]) == int(after_meta["predicted_index"]):
            continue
        pairs.append(
            {
                "method": method,
                "sample_id": sample_id,
                "before": before_meta,
                "after": after_meta,
            }
        )
    return pairs


def state_order(method: str, sample_id: str) -> Tuple[str, str]:
    digest = hashlib.sha256(f"paired-supervision::{method}::{sample_id}".encode()).digest()
    return ("before", "after") if digest[0] % 2 == 0 else ("after", "before")


def build_prompt(condition: str, class_a: str, class_b: str) -> str:
    common = (
        "You are supervising two anonymized states of the same prototype-based bird classifier. "
        "Choose which state has the more trustworthy prediction. The states are randomly ordered; "
        "neither A nor B should be assumed newer or better. The ground-truth label is withheld.\n\n"
        f"STATE A predicts: {class_a}\n"
        f"STATE B predicts: {class_b}\n\n"
    )
    if condition == "predictions_only":
        evidence = (
            "No image or prototype evidence is provided. Base the choice only on the two prediction names; "
            "this is intentionally a weak control condition.\n\n"
        )
    elif condition == "image_predictions":
        evidence = (
            "IMAGE 1 is the corrupted test image. Use visible bird appearance and the two prediction names. "
            "No classifier reasoning evidence is provided.\n\n"
        )
    else:
        evidence = (
            "You receive five images in order:\n"
            "IMAGE 1: corrupted test image.\n"
            "IMAGE 2: STATE A predicted-class reasoning board.\n"
            "IMAGE 3: STATE A any-class reasoning board.\n"
            "IMAGE 4: STATE B predicted-class reasoning board.\n"
            "IMAGE 5: STATE B any-class reasoning board.\n"
            "Each predicted-class board shows the localized input region and retrieved training prototypes. "
            "Each any-class board exposes strong competing-class prototypes. Compare localization, visual "
            "prototype match, class consistency, and spurious evidence.\n\n"
        )
    return (
        common
        + evidence
        + "Which state is more likely to be correct? Your entire response must be one character: A or B. "
        "Do not explain.\nANSWER:"
    )


def ensure_blind_boards(sample_dir: Path) -> Tuple[Path, Path]:
    predicted = sample_dir / PREDICTED_BOARD_NAME
    any_class = sample_dir / ANY_CLASS_BOARD_NAME
    if not predicted.exists():
        sanitize_board(
            sample_dir / "01_predicted_class_reasoning.png",
            predicted,
            "predicted-class reasoning board",
        )
    if not any_class.exists():
        sanitize_board(
            sample_dir / "02_any_class_reasoning.png",
            any_class,
            "any-class reasoning board",
        )
    return predicted, any_class


def condition_images(
    results_dir: Path, pair: Dict, condition: str, order: Tuple[str, str]
) -> List[Path]:
    before_dir = results_dir / "unadapted" / "samples" / pair["sample_id"]
    if condition == "predictions_only":
        return []
    raw = before_dir / "00_corrupted_input.png"
    if condition == "image_predictions":
        return [raw]

    state_dirs = {
        "before": before_dir,
        "after": results_dir / pair["method"] / "samples" / pair["sample_id"],
    }
    board_a = ensure_blind_boards(state_dirs[order[0]])
    board_b = ensure_blind_boards(state_dirs[order[1]])
    return [raw, board_a[0], board_a[1], board_b[0], board_b[1]]


def choice_probability(scorer, image_paths: Sequence[Path], prompt: str) -> Tuple[str, float, float]:
    import torch

    inputs = scorer._prepare_inputs(image_paths, prompt)
    tokenizer = scorer.processor.tokenizer
    a_ids = [tokenizer.encode(text, add_special_tokens=False) for text in ("A", " A")]
    b_ids = [tokenizer.encode(text, add_special_tokens=False) for text in ("B", " B")]
    if not all(len(ids) == 1 for ids in a_ids + b_ids):
        raise RuntimeError("A/B choices are not single tokens for this tokenizer")
    with torch.inference_mode():
        output = scorer.model(
            **inputs,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    final_logits = output.logits[0, -1].float()
    logit_a = torch.logsumexp(final_logits[[ids[0] for ids in a_ids]], dim=0)
    logit_b = torch.logsumexp(final_logits[[ids[0] for ids in b_ids]], dim=0)
    probabilities = torch.softmax(torch.stack([logit_a, logit_b]), dim=0)
    probability_a = float(probabilities[0].item())
    probability_b = float(probabilities[1].item())
    return ("A" if probability_a >= probability_b else "B"), probability_a, probability_b


def score_path(output_dir: Path, method: str, condition: str, sample_id: str) -> Path:
    return output_dir / "scores" / method / condition / f"{sample_id}.json"


def run_score(args: argparse.Namespace) -> None:
    from vlm_eval import VLMScorer

    pairs = changed_pairs(args.results_dir, args.method)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    scorer = VLMScorer(args.vlm_model_id, max_new_tokens=1)
    completed = skipped = failed = 0
    for pair in pairs:
        output_path = score_path(args.output_dir, args.method, args.condition, pair["sample_id"])
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        order = state_order(args.method, pair["sample_id"])
        states = {"before": pair["before"], "after": pair["after"]}
        class_a = str(states[order[0]]["predicted_class"])
        class_b = str(states[order[1]]["predicted_class"])
        prompt = build_prompt(args.condition, class_a, class_b)
        images = condition_images(args.results_dir, pair, args.condition, order)
        try:
            preferred, probability_a, probability_b = choice_probability(scorer, images, prompt)
            preferred_role = order[0] if preferred == "A" else order[1]
            adapted_probability = probability_a if order[0] == "after" else probability_b
            write_json(
                output_path,
                {
                    "schema_version": 1,
                    "label_blind": True,
                    "method_blind": True,
                    "randomized_state_order": True,
                    "method": args.method,
                    "condition": args.condition,
                    "sample_id": pair["sample_id"],
                    "state_a_role": order[0],
                    "state_b_role": order[1],
                    "state_a_prediction": class_a,
                    "state_b_prediction": class_b,
                    "preferred_state": preferred,
                    "preferred_role": preferred_role,
                    "probability_a": probability_a,
                    "probability_b": probability_b,
                    "adapted_preference_probability": adapted_probability,
                    "vlm_model_id": args.vlm_model_id,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                },
            )
            completed += 1
            LOGGER.info(
                "Scored %s/%s/%s: prefer=%s p(adapted)=%.3f",
                args.method,
                args.condition,
                pair["sample_id"],
                preferred_role,
                adapted_probability,
            )
        except Exception as exc:
            failed += 1
            LOGGER.exception("Failed %s/%s/%s", args.method, args.condition, pair["sample_id"])
            error_path = output_path.with_suffix(".error.txt")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(str(exc), encoding="utf-8")
    LOGGER.info("Finished: completed=%d skipped=%d failed=%d", completed, skipped, failed)


def outcome(before: Dict, after: Dict) -> str:
    before_correct = bool(before["is_correct"])
    after_correct = bool(after["is_correct"])
    if not before_correct and after_correct:
        return "improved"
    if before_correct and not after_correct:
        return "harmed"
    return "neither_correct" if not before_correct else "both_correct"


def collect_analysis_records(results_dir: Path, output_dir: Path) -> List[Dict]:
    records = []
    for method in METHOD_ORDER:
        for pair in changed_pairs(results_dir, method):
            scores = {}
            for condition in CONDITION_ORDER:
                path = score_path(output_dir, method, condition, pair["sample_id"])
                if path.exists():
                    scores[condition] = read_json(path)
            if len(scores) != len(CONDITION_ORDER):
                continue
            record = {
                "pair_id": f"{method}::{pair['sample_id']}",
                "method": method,
                "sample_id": pair["sample_id"],
                "outcome": outcome(pair["before"], pair["after"]),
                "before_correct": int(bool(pair["before"]["is_correct"])),
                "after_correct": int(bool(pair["after"]["is_correct"])),
                "before_prediction": pair["before"]["predicted_class"],
                "after_prediction": pair["after"]["predicted_class"],
                "ground_truth": pair["before"]["ground_truth_class"],
            }
            for condition, score in scores.items():
                chose_after = score["preferred_role"] == "after"
                record[f"{condition}_chose_after"] = int(chose_after)
                record[f"{condition}_adapted_probability"] = float(
                    score["adapted_preference_probability"]
                )
                record[f"{condition}_chosen_correct"] = int(
                    pair["after"]["is_correct"] if chose_after else pair["before"]["is_correct"]
                )
            records.append(record)
    return records


def metric_summary(records: Sequence[Dict], condition: str) -> Dict:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    decisive = [record for record in records if record["outcome"] in {"improved", "harmed"}]
    target = np.asarray([record["after_correct"] for record in decisive], dtype=int)
    choice = np.asarray([record[f"{condition}_chose_after"] for record in decisive], dtype=int)
    probability = np.asarray(
        [record[f"{condition}_adapted_probability"] for record in decisive], dtype=float
    )
    return {
        "n_changed": len(records),
        "n_decisive": len(decisive),
        "selection_accuracy": float(np.mean(choice == target)),
        "balanced_selection_accuracy": float(balanced_accuracy_score(target, choice)),
        "adaptation_preference_auroc": float(roc_auc_score(target, probability)),
        "adaptation_preference_aupr": float(average_precision_score(target, probability)),
        "changed_set_accuracy": float(
            np.mean([record[f"{condition}_chosen_correct"] for record in records])
        ),
        "num_accept": int(sum(record[f"{condition}_chose_after"] for record in records)),
    }


def all_sample_controller_summary(
    results_dir: Path, records: Sequence[Dict], condition: str
) -> Dict:
    choices = {record["pair_id"]: bool(record[f"{condition}_chose_after"]) for record in records}
    before_maps = sample_meta_map(results_dir, "unadapted")
    outcomes = []
    for method in METHOD_ORDER:
        after_map = sample_meta_map(results_dir, method)
        for sample_id in sorted(set(before_maps) & set(after_map)):
            before = before_maps[sample_id]
            after = after_map[sample_id]
            pair_id = f"{method}::{sample_id}"
            changed = int(before["predicted_index"]) != int(after["predicted_index"])
            if changed and pair_id not in choices:
                continue
            chose_after = choices.get(pair_id, True)
            outcomes.append(
                {
                    "controller_correct": bool(after["is_correct"] if chose_after else before["is_correct"]),
                    "before_correct": bool(before["is_correct"]),
                    "after_correct": bool(after["is_correct"]),
                    "oracle_correct": bool(before["is_correct"] or after["is_correct"]),
                }
            )
    return {
        "n": len(outcomes),
        "controller_accuracy": float(np.mean([item["controller_correct"] for item in outcomes])),
        "always_before_accuracy": float(np.mean([item["before_correct"] for item in outcomes])),
        "always_after_accuracy": float(np.mean([item["after_correct"] for item in outcomes])),
        "oracle_accuracy": float(np.mean([item["oracle_correct"] for item in outcomes])),
    }


def paired_comparison(
    records: Sequence[Dict], left: str, right: str, resamples: int, seed: int
) -> Dict:
    from scipy.stats import binomtest

    decisive = [record for record in records if record["outcome"] in {"improved", "harmed"}]
    left_correct = np.asarray(
        [record[f"{left}_chosen_correct"] for record in decisive], dtype=int
    )
    right_correct = np.asarray(
        [record[f"{right}_chosen_correct"] for record in decisive], dtype=int
    )
    point = float(np.mean(left_correct - right_correct))
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(decisive):
        groups.setdefault(record["sample_id"], []).append(index)
    names = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.asarray([index for name in sampled for index in groups[str(name)]], dtype=int)
        draws.append(float(np.mean(left_correct[indices] - right_correct[indices])))
    left_only = int(np.sum((left_correct == 1) & (right_correct == 0)))
    right_only = int(np.sum((left_correct == 0) & (right_correct == 1)))
    discordant = left_only + right_only
    p_value = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "left": left,
        "right": right,
        "selection_accuracy_difference": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "mcnemar_exact_p": p_value,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
    }


def write_csv(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_markdown(payload: Dict) -> str:
    lines = [
        "# Paired VLM Adaptation Supervision",
        "",
        "The VLM compared randomly ordered, anonymized classifier states A/B. Ground truth and before/after identity were hidden during scoring.",
        "Only changed predictions were sent to the VLM; the decisive subset contains cases where exactly one state was correct.",
        "",
        "## Decisive state-selection result",
        "",
        "Condition | N | Selection Acc. | Balanced Acc. | AUROC | AUPR",
        "--------- | - | -------------- | ------------- | ----- | ----",
    ]
    for condition in CONDITION_ORDER:
        item = payload["conditions"][condition]
        lines.append(
            f"{CONDITION_NAMES[condition]} | {item['n_decisive']} | {item['selection_accuracy']:.3f} | "
            f"{item['balanced_selection_accuracy']:.3f} | {item['adaptation_preference_auroc']:.3f} | "
            f"{item['adaptation_preference_aupr']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Practical controller accuracy",
            "",
            "Condition | N | Controller | Always before | Always adapted | Oracle",
            "--------- | - | ---------- | ------------- | -------------- | ------",
        ]
    )
    for condition in CONDITION_ORDER:
        item = payload["controller"][condition]
        lines.append(
            f"{CONDITION_NAMES[condition]} | {item['n']} | {item['controller_accuracy']:.3f} | "
            f"{item['always_before_accuracy']:.3f} | {item['always_after_accuracy']:.3f} | "
            f"{item['oracle_accuracy']:.3f}"
        )
    lines.extend(["", "## Paired board-ablation tests", ""])
    for comparison in payload["comparisons"]:
        low, high = comparison["ci95"]
        lines.append(
            f"- `{comparison['left']} - {comparison['right']}`: "
            f"delta selection accuracy={comparison['selection_accuracy_difference']:+.3f} "
            f"(95% CI [{low:+.3f}, {high:+.3f}]), exact McNemar p={comparison['mcnemar_exact_p']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Changed prediction pairs available: {payload['coverage']['available_changed_pairs']}",
            f"- Complete across all three conditions: {payload['coverage']['complete_pairs']}",
            f"- Decisive pairs (one state correct): {payload['coverage']['decisive_pairs']}",
            f"- Improved by adaptation: {payload['coverage']['improved_pairs']}",
            f"- Harmed by adaptation: {payload['coverage']['harmed_pairs']}",
            "",
            "The true label was used only after VLM scoring to evaluate the supervisor.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_results(path: Path, payload: Dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [CONDITION_NAMES[condition] for condition in CONDITION_ORDER]
    selection = [payload["conditions"][condition]["selection_accuracy"] for condition in CONDITION_ORDER]
    controller = [payload["controller"][condition]["controller_accuracy"] for condition in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#9C9C9C", "#4C78A8", "#54A24B"]
    axes[0].bar(labels, selection, color=colors)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Selection accuracy")
    axes[0].set_title("Choose the correct state (decisive pairs)")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, controller, color=colors)
    axes[1].axhline(
        payload["controller"]["full_reasoning"]["always_after_accuracy"],
        color="#E45756",
        linestyle="--",
        label="Always adapted",
    )
    axes[1].set_ylabel("Classification accuracy")
    axes[1].set_title("VLM-gated adaptation")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analyze(args: argparse.Namespace) -> None:
    records = collect_analysis_records(args.results_dir, args.output_dir)
    if not records:
        raise SystemExit("No complete paired-supervision records found")
    available = sum(len(changed_pairs(args.results_dir, method)) for method in METHOD_ORDER)
    decisive = [record for record in records if record["outcome"] in {"improved", "harmed"}]
    payload = {
        "schema_version": 1,
        "vlm_input_labels": "before and after predicted class names",
        "ground_truth_visible_to_vlm": False,
        "state_order_randomized": True,
        "conditions": {
            condition: metric_summary(records, condition) for condition in CONDITION_ORDER
        },
        "controller": {
            condition: all_sample_controller_summary(args.results_dir, records, condition)
            for condition in CONDITION_ORDER
        },
        "comparisons": [
            paired_comparison(
                records,
                "full_reasoning",
                "image_predictions",
                args.bootstrap_resamples,
                args.seed,
            ),
            paired_comparison(
                records,
                "full_reasoning",
                "predictions_only",
                args.bootstrap_resamples,
                args.seed + 1,
            ),
        ],
        "coverage": {
            "available_changed_pairs": available,
            "complete_pairs": len(records),
            "decisive_pairs": len(decisive),
            "improved_pairs": sum(record["outcome"] == "improved" for record in records),
            "harmed_pairs": sum(record["outcome"] == "harmed" for record in records),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "paired_supervision_metrics.json", payload)
    write_csv(args.output_dir / "paired_supervision_records.csv", records)
    markdown = build_markdown(payload)
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    plot_results(args.output_dir / "paired_supervision_results.png", payload)
    print(markdown)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.command == "score":
        run_score(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
