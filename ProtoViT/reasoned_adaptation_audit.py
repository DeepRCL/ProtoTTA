#!/usr/bin/env python3
"""Long-form VLM audit of continuous test-time adaptation.

The VLM is explicitly shown BEFORE and AFTER continuous adaptation.  Ground
truth and correctness are withheld until evaluation.  Three matched conditions
measure whether prototype reasoning boards add information beyond method/prediction
priors and ordinary visual recognition:

* predictions_only: method name plus before/after predicted labels
* image_predictions: corrupted image plus the same text
* full_reasoning: corrupted image plus a clearly labelled before/after board

Only decisive pairs (exactly one state is correct) require a VLM decision.  This
is the relevant subset for accepting versus rolling back adaptation; decisions
on pairs where both states are correct or both are wrong cannot change accuracy.
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
from PIL import Image, ImageDraw, ImageFont

from failure_detection import ANY_CLASS_BOARD_NAME, PREDICTED_BOARD_NAME, sanitize_board


METHOD_ORDER = ["tent", "eata", "prototta"]
CONDITION_ORDER = ["predictions_only", "image_predictions", "full_reasoning"]
DISPLAY_NAMES = {"tent": "Tent", "eata": "EATA", "prototta": "ProtoTTA"}
CONDITION_NAMES = {
    "predictions_only": "Predictions only",
    "image_predictions": "Image + predictions",
    "full_reasoning": "Image + paired reasoning boards",
}
LOGGER = logging.getLogger("reasoned_adaptation_audit")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=script_dir / "results" / "vlm_eval")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "vlm_eval" / "reasoned_adaptation_audit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create and validate paired evidence canvases.")
    prepare.add_argument("--method", choices=METHOD_ORDER + ["all"], default="all")

    score = subparsers.add_parser("score", help="Generate long-form VLM audits.")
    score.add_argument("--method", choices=METHOD_ORDER, required=True)
    score.add_argument("--condition", choices=CONDITION_ORDER, required=True)
    score.add_argument("--vlm-model-id", default="Qwen/Qwen3.6-35B-A3B")
    score.add_argument("--max-new-tokens", type=int, default=2048)
    score.add_argument(
        "--non-thinking",
        action="store_true",
        help="Disable Qwen thinking through its chat template (recommended for Qwen3.6).",
    )
    score.add_argument("--max-samples", type=int, default=None)
    score.add_argument("--overwrite", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Evaluate supervision and board ablations.")
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
    if not root.exists():
        return {}
    return {
        path.parent.name: read_json(path)
        for path in sorted(root.glob("*/03_meta.json"))
    }


def pair_outcome(before: Dict, after: Dict) -> str:
    before_correct = bool(before["is_correct"])
    after_correct = bool(after["is_correct"])
    if not before_correct and after_correct:
        return "improved"
    if before_correct and not after_correct:
        return "harmed"
    return "both_correct" if before_correct else "neither_correct"


def all_pairs(results_dir: Path, method: str) -> List[Dict]:
    before_map = sample_meta_map(results_dir, "unadapted")
    after_map = sample_meta_map(results_dir, method)
    pairs = []
    for sample_id in sorted(set(before_map) & set(after_map)):
        before = before_map[sample_id]
        after = after_map[sample_id]
        pairs.append(
            {
                "method": method,
                "sample_id": sample_id,
                "before": before,
                "after": after,
                "outcome": pair_outcome(before, after),
            }
        )
    return pairs


def decisive_pairs(results_dir: Path, method: str) -> List[Dict]:
    return [pair for pair in all_pairs(results_dir, method) if pair["outcome"] in {"improved", "harmed"}]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def ensure_blind_boards(sample_dir: Path) -> Tuple[Path, Path]:
    predicted = sample_dir / PREDICTED_BOARD_NAME
    any_class = sample_dir / ANY_CLASS_BOARD_NAME
    if not predicted.exists():
        sanitize_board(sample_dir / "01_predicted_class_reasoning.png", predicted, "predicted-class reasoning board")
    if not any_class.exists():
        sanitize_board(sample_dir / "02_any_class_reasoning.png", any_class, "any-class reasoning board")
    return predicted, any_class


def _fit_image(path: Path, width: int) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def canvas_path(output_dir: Path, method: str, sample_id: str) -> Path:
    return output_dir / "canvases" / method / f"{sample_id}.jpg"


def build_evidence_canvas(results_dir: Path, output_dir: Path, pair: Dict) -> Path:
    """Build an explicit two-column comparison without ground-truth leakage."""
    destination = canvas_path(output_dir, pair["method"], pair["sample_id"])
    if destination.exists():
        return destination
    before_dir = results_dir / "unadapted" / "samples" / pair["sample_id"]
    after_dir = results_dir / pair["method"] / "samples" / pair["sample_id"]
    before_boards = ensure_blind_boards(before_dir)
    after_boards = ensure_blind_boards(after_dir)

    column_width = 1350
    margin = 35
    gutter = 50
    header_height = 190
    section_height = 52
    board_width = column_width - 2 * margin
    board_sets = [
        [_fit_image(path, board_width) for path in before_boards],
        [_fit_image(path, board_width) for path in after_boards],
    ]
    content_heights = [sum(image.height for image in images) + 2 * section_height + margin for images in board_sets]
    total_width = 2 * column_width + gutter
    total_height = header_height + max(content_heights)
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    subtitle_font = _font(30, bold=True)
    section_font = _font(27, bold=True)
    small_font = _font(24)
    colors = ["#285A8E", "#A54822"]
    roles = ["BEFORE ADAPTATION", "AFTER CONTINUOUS ADAPTATION"]
    predictions = [pair["before"]["predicted_class"], pair["after"]["predicted_class"]]

    for column, (role, prediction, images, color) in enumerate(zip(roles, predictions, board_sets, colors)):
        x0 = column * (column_width + gutter)
        draw.rectangle((x0, 0, x0 + column_width, header_height), fill=color)
        draw.text((x0 + margin, 24), role, fill="white", font=title_font)
        draw.text((x0 + margin, 88), f"Prediction: {prediction}", fill="white", font=subtitle_font)
        draw.text((x0 + margin, 139), "Ground truth and correctness withheld", fill="white", font=small_font)
        y = header_height + 8
        for label, image in zip(("PREDICTED-CLASS EVIDENCE", "ANY-CLASS / COMPETING EVIDENCE"), images):
            draw.rectangle((x0, y, x0 + column_width, y + section_height), fill="#E9EEF3")
            draw.text((x0 + margin, y + 9), label, fill="#202020", font=section_font)
            y += section_height
            canvas.paste(image, (x0 + margin, y))
            y += image.height
    draw.rectangle((column_width, 0, column_width + gutter, total_height), fill="#222222")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, optimize=True)
    return destination


def build_prompt(method: str, condition: str, before_prediction: str, after_prediction: str) -> str:
    common = (
        "You are an independent supervisor auditing a prototype-based bird classifier. "
        "Decide whether its continuous test-time adaptation should be ACCEPTED for this sample or ROLLED BACK.\n\n"
        "Temporal semantics are explicit:\n"
        "- BEFORE is the frozen, unadapted source model.\n"
        f"- AFTER is {DISPLAY_NAMES[method]} after online CONTINUOUS adaptation in the corruption stream. "
        "Its parameters/state accumulated from preceding test batches; it was not reset for this image.\n"
        "- Adaptation is not assumed beneficial: AFTER may be better or worse.\n\n"
        f"BEFORE prediction: {before_prediction}\n"
        f"AFTER prediction: {after_prediction}\n"
        "The true class and correctness of both predictions are deliberately withheld. Never infer correctness from the words BEFORE/AFTER.\n\n"
    )
    if condition == "predictions_only":
        evidence = (
            "CONTROL CONDITION: no test image and no prototype evidence are available. Judge only from the method and prediction names. "
            "State clearly that visual changes are not observable.\n\n"
        )
    elif condition == "image_predictions":
        evidence = (
            "IMAGE 1 is the corrupted test image. Compare the two predicted species against visible bird appearance. "
            "No classifier reasoning boards are available, so do not claim to observe changes in model attention or prototypes.\n\n"
        )
    else:
        evidence = (
            "You receive two images:\n"
            "IMAGE 1 is the corrupted test image.\n"
            "IMAGE 2 is one explicit side-by-side evidence canvas: LEFT/blue is BEFORE and RIGHT/orange is AFTER continuous adaptation. "
            "Each side contains (i) a predicted-class board with the input, strongest-match heatmap, and five retrieved training prototype patches, "
            "and (ii) an any-class board exposing ten strong prototypes including competitors. Prototype patches are stored training exemplars, not crops from this test image.\n\n"
            "Compare BEFORE versus AFTER in this order: (1) whether the heatmap moves toward a real, discriminative bird part instead of background/noise; "
            "(2) whether retrieved prototype patches visually and anatomically support the stated prediction; "
            "(3) whether competing-class evidence becomes less spurious or more coherent; and (4) whether the change supports AFTER more strongly than BEFORE.\n\n"
        )
    schema = (
        "First reason carefully using only the supplied evidence, then return exactly one JSON object with these keys:\n"
        '- "before_evidence_quality": integer 1-5;\n'
        '- "after_evidence_quality": integer 1-5;\n'
        '- "adaptation_score": integer from -5 (clearly harmful) through 0 (no reliable benefit) to +5 (clearly beneficial);\n'
        '- "adaptation_quality": exactly "IMPROVED", "HARMED", or "NO_MEANINGFUL_CHANGE";\n'
        '- "recommended_action": exactly "ACCEPT" or "ROLLBACK";\n'
        '- "focus_change": one specific sentence, or say it is not observable;\n'
        '- "prototype_change": one specific sentence, or say it is not observable;\n'
        '- "comparative_analysis": 3-6 evidence-grounded sentences explaining why AFTER is or is not more trustworthy;\n'
        '- "confidence": integer 1-5.\n\n'
        "ACCEPT only if AFTER is more likely correct than BEFORE. Otherwise ROLLBACK. Do not mention or guess a hidden ground-truth label. "
        "Return no markdown and no text outside the JSON object."
    )
    return common + evidence + schema


def validate_audit(payload: Dict) -> Dict:
    cleaned = dict(payload)
    for key in ("before_evidence_quality", "after_evidence_quality", "confidence"):
        value = int(payload[key])
        if not 1 <= value <= 5:
            raise ValueError(f"{key} must be in [1,5], received {value}")
        cleaned[key] = value
    score = int(payload["adaptation_score"])
    if not -5 <= score <= 5:
        raise ValueError(f"adaptation_score must be in [-5,5], received {score}")
    cleaned["adaptation_score"] = score
    quality = str(payload["adaptation_quality"]).upper().strip()
    action = str(payload["recommended_action"]).upper().strip()
    if quality not in {"IMPROVED", "HARMED", "NO_MEANINGFUL_CHANGE"}:
        raise ValueError(f"invalid adaptation_quality: {quality}")
    if action not in {"ACCEPT", "ROLLBACK"}:
        raise ValueError(f"invalid recommended_action: {action}")
    cleaned["adaptation_quality"] = quality
    cleaned["recommended_action"] = action
    for key in ("focus_change", "prototype_change", "comparative_analysis"):
        value = str(payload.get(key, "")).strip()
        if not value:
            raise ValueError(f"{key} is empty")
        cleaned[key] = value
    return cleaned


def score_path(output_dir: Path, method: str, condition: str, sample_id: str) -> Path:
    return output_dir / "scores" / method / condition / f"{sample_id}.json"


def condition_images(results_dir: Path, output_dir: Path, pair: Dict, condition: str) -> List[Path]:
    if condition == "predictions_only":
        return []
    raw = results_dir / "unadapted" / "samples" / pair["sample_id"] / "00_corrupted_input.png"
    if condition == "image_predictions":
        return [raw]
    return [raw, build_evidence_canvas(results_dir, output_dir, pair)]


def run_prepare(args: argparse.Namespace) -> None:
    methods = METHOD_ORDER if args.method == "all" else [args.method]
    created = 0
    for method in methods:
        for pair in decisive_pairs(args.results_dir, method):
            path = build_evidence_canvas(args.results_dir, args.output_dir, pair)
            with Image.open(path) as image:
                if image.width < 2000 or image.height < 500:
                    raise RuntimeError(f"Invalid evidence canvas dimensions: {path} {image.size}")
            created += 1
    print(f"Validated {created} decisive-pair canvases")


def run_score(args: argparse.Namespace) -> None:
    from vlm_eval import VLMScorer

    pairs = decisive_pairs(args.results_dir, args.method)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    scorer = VLMScorer(
        args.vlm_model_id,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=False if args.non_thinking else None,
    )
    completed = skipped = failed = 0
    for pair in pairs:
        output_path = score_path(args.output_dir, args.method, args.condition, pair["sample_id"])
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        prompt = build_prompt(
            args.method,
            args.condition,
            str(pair["before"]["predicted_class"]),
            str(pair["after"]["predicted_class"]),
        )
        images = condition_images(args.results_dir, args.output_dir, pair, args.condition)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            parsed, raw_text, actual_prompt = scorer.generate_json(images, prompt)
            audit = validate_audit(parsed)
            output_path.with_suffix(".raw.txt").write_text(raw_text, encoding="utf-8")
            write_json(
                output_path,
                {
                    "schema_version": 1,
                    "method": args.method,
                    "condition": args.condition,
                    "sample_id": pair["sample_id"],
                    "before_prediction": pair["before"]["predicted_class"],
                    "after_prediction": pair["after"]["predicted_class"],
                    "ground_truth_visible_to_vlm": False,
                    "before_after_explicit": True,
                    "continuous_adaptation": True,
                    "episodic_reset": False,
                    "vlm_model_id": args.vlm_model_id,
                    "max_new_tokens": args.max_new_tokens,
                    "thinking_enabled": not args.non_thinking,
                    "prompt_sha256": hashlib.sha256(actual_prompt.encode("utf-8")).hexdigest(),
                    **audit,
                    "raw_response": raw_text,
                },
            )
            output_path.with_suffix(".error.txt").unlink(missing_ok=True)
            completed += 1
            LOGGER.info(
                "Audited %s/%s/%s: %s score=%+d",
                args.method,
                args.condition,
                pair["sample_id"],
                audit["recommended_action"],
                audit["adaptation_score"],
            )
        except Exception as exc:
            failed += 1
            LOGGER.exception("Audit failed for %s/%s/%s", args.method, args.condition, pair["sample_id"])
            raw = getattr(exc, "raw_text", "")
            if raw:
                output_path.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
            output_path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")
    LOGGER.info("Finished: completed=%d skipped=%d failed=%d", completed, skipped, failed)


def collect_records(results_dir: Path, output_dir: Path) -> List[Dict]:
    records = []
    for method in METHOD_ORDER:
        for pair in decisive_pairs(results_dir, method):
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
                "outcome": pair["outcome"],
                "after_correct": int(bool(pair["after"]["is_correct"])),
                "before_prediction": pair["before"]["predicted_class"],
                "after_prediction": pair["after"]["predicted_class"],
                "ground_truth": pair["before"]["ground_truth_class"],
            }
            for condition, score in scores.items():
                chose_after = str(score["recommended_action"]) == "ACCEPT"
                record[f"{condition}_chose_after"] = int(chose_after)
                record[f"{condition}_adaptation_score"] = int(score["adaptation_score"])
                record[f"{condition}_confidence"] = int(score["confidence"])
                record[f"{condition}_chosen_correct"] = int(chose_after == bool(pair["after"]["is_correct"]))
                record[f"{condition}_decision_score_inconsistent"] = int(
                    (chose_after and int(score["adaptation_score"]) <= 0)
                    or (not chose_after and int(score["adaptation_score"]) > 0)
                )
            records.append(record)
    return records


def condition_metrics(records: Sequence[Dict], condition: str) -> Dict:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    target = np.asarray([record["after_correct"] for record in records], dtype=int)
    decision = np.asarray([record[f"{condition}_chose_after"] for record in records], dtype=int)
    score = np.asarray([record[f"{condition}_adaptation_score"] for record in records], dtype=float)
    return {
        "n": len(records),
        "selection_accuracy": float(np.mean(decision == target)),
        "balanced_selection_accuracy": float(balanced_accuracy_score(target, decision)),
        "adaptation_score_auroc": float(roc_auc_score(target, score)),
        "adaptation_score_aupr": float(average_precision_score(target, score)),
        "accept_rate": float(np.mean(decision)),
        "mean_adaptation_score": float(np.mean(score)),
        "decision_score_inconsistencies": int(sum(record[f"{condition}_decision_score_inconsistent"] for record in records)),
    }


def controller_metrics(results_dir: Path, records: Sequence[Dict], condition: str) -> Dict:
    choices = {record["pair_id"]: bool(record[f"{condition}_chose_after"]) for record in records}
    rows = []
    for method in METHOD_ORDER:
        for pair in all_pairs(results_dir, method):
            pair_id = f"{method}::{pair['sample_id']}"
            if pair["outcome"] in {"improved", "harmed"}:
                if pair_id not in choices:
                    continue
                controller_correct = bool(pair["after"]["is_correct"]) if choices[pair_id] else bool(pair["before"]["is_correct"])
            else:
                # The choice cannot affect correctness when the two states agree in correctness.
                controller_correct = bool(pair["before"]["is_correct"])
            rows.append(
                {
                    "controller": controller_correct,
                    "before": bool(pair["before"]["is_correct"]),
                    "after": bool(pair["after"]["is_correct"]),
                    "oracle": bool(pair["before"]["is_correct"] or pair["after"]["is_correct"]),
                }
            )
    return {
        "n": len(rows),
        "controller_accuracy": float(np.mean([row["controller"] for row in rows])),
        "always_before_accuracy": float(np.mean([row["before"] for row in rows])),
        "always_after_accuracy": float(np.mean([row["after"] for row in rows])),
        "oracle_accuracy": float(np.mean([row["oracle"] for row in rows])),
    }


def paired_bootstrap(records: Sequence[Dict], left: str, right: str, resamples: int, seed: int) -> Dict:
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score

    target = np.asarray([record["after_correct"] for record in records], dtype=int)
    left_decision = np.asarray([record[f"{left}_chose_after"] for record in records], dtype=int)
    right_decision = np.asarray([record[f"{right}_chose_after"] for record in records], dtype=int)
    left_score = np.asarray([record[f"{left}_adaptation_score"] for record in records], dtype=float)
    right_score = np.asarray([record[f"{right}_adaptation_score"] for record in records], dtype=float)
    accuracy_delta = float(np.mean(left_decision == target) - np.mean(right_decision == target))
    auroc_delta = float(roc_auc_score(target, left_score) - roc_auc_score(target, right_score))

    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["sample_id"], []).append(index)
    names = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    accuracy_draws, auroc_draws = [], []
    for _ in range(resamples):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.asarray([index for name in sampled for index in groups[str(name)]], dtype=int)
        sampled_target = target[indices]
        accuracy_draws.append(float(np.mean(left_decision[indices] == sampled_target) - np.mean(right_decision[indices] == sampled_target)))
        if len(np.unique(sampled_target)) == 2:
            auroc_draws.append(float(roc_auc_score(sampled_target, left_score[indices]) - roc_auc_score(sampled_target, right_score[indices])))

    left_correct = left_decision == target
    right_correct = right_decision == target
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    return {
        "left": left,
        "right": right,
        "selection_accuracy_difference": accuracy_delta,
        "selection_accuracy_ci95": [float(np.quantile(accuracy_draws, 0.025)), float(np.quantile(accuracy_draws, 0.975))],
        "adaptation_score_auroc_difference": auroc_delta,
        "adaptation_score_auroc_ci95": [float(np.quantile(auroc_draws, 0.025)), float(np.quantile(auroc_draws, 0.975))],
        "mcnemar_exact_p": float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0,
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
        "# Reasoned Continuous-Adaptation Audit",
        "",
        "Qwen3.6-35B-A3B (non-thinking/instruct mode) was explicitly told which state was BEFORE and which was AFTER continuous, non-reset adaptation. It generated a comparative audit before recommending ACCEPT or ROLLBACK. Ground truth and correctness were hidden until evaluation.",
        "",
        "## Correct-state selection on decisive pairs",
        "",
        "Condition | N | Selection acc. | Balanced acc. | Score AUROC | Score AUPR | Accept rate",
        "--------- | - | -------------- | ------------- | ----------- | ---------- | -----------",
    ]
    for condition in CONDITION_ORDER:
        item = payload["conditions"][condition]
        lines.append(
            f"{CONDITION_NAMES[condition]} | {item['n']} | {item['selection_accuracy']:.3f} | "
            f"{item['balanced_selection_accuracy']:.3f} | {item['adaptation_score_auroc']:.3f} | "
            f"{item['adaptation_score_aupr']:.3f} | {item['accept_rate']:.3f}"
        )
    lines.extend([
        "",
        "Here AUROC asks whether the continuous adaptation score ranks truly improved cases above truly harmed cases. Selection accuracy tests the actual accept/rollback recommendation.",
        "",
        "## VLM-gated classifier accuracy on the full 300 method-sample pairs",
        "",
        "Condition | Controller | Always before | Always adapted | Oracle",
        "--------- | ---------- | ------------- | -------------- | ------",
    ])
    for condition in CONDITION_ORDER:
        item = payload["controller"][condition]
        lines.append(
            f"{CONDITION_NAMES[condition]} | {item['controller_accuracy']:.3f} | {item['always_before_accuracy']:.3f} | "
            f"{item['always_after_accuracy']:.3f} | {item['oracle_accuracy']:.3f}"
        )
    lines.extend(["", "## Paired board-ablation tests", ""])
    for item in payload["comparisons"]:
        acc_low, acc_high = item["selection_accuracy_ci95"]
        auc_low, auc_high = item["adaptation_score_auroc_ci95"]
        lines.append(
            f"- `{item['left']} - {item['right']}`: selection delta {item['selection_accuracy_difference']:+.3f} "
            f"(95% CI [{acc_low:+.3f}, {acc_high:+.3f}], McNemar p={item['mcnemar_exact_p']:.4f}); "
            f"AUROC delta {item['adaptation_score_auroc_difference']:+.3f} (95% CI [{auc_low:+.3f}, {auc_high:+.3f}])."
        )
    coverage = payload["coverage"]
    lines.extend([
        "",
        "## Coverage and protocol",
        "",
        f"- Complete decisive pairs: {coverage['complete_decisive_pairs']} / {coverage['available_decisive_pairs']} (improved={coverage['improved_pairs']}, harmed={coverage['harmed_pairs']}).",
        "- All conditions receive identical temporal/method/prediction text. Only the evidence modality changes.",
        "- The paired canvas uses sanitized boards; no ground-truth class, CORRECT/WRONG marker, or outcome appears in the VLM input.",
        "- The classifier artifacts were produced with EPISODIC=False; each method was updated sequentially through a corruption stream. ProtoTTA used no reset.",
        "- Non-decisive pairs do not need a VLM call because choosing either state cannot change correctness.",
    ])
    return "\n".join(lines) + "\n"


def plot_results(path: Path, payload: Dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [CONDITION_NAMES[condition] for condition in CONDITION_ORDER]
    selection = [payload["conditions"][condition]["selection_accuracy"] for condition in CONDITION_ORDER]
    auroc = [payload["conditions"][condition]["adaptation_score_auroc"] for condition in CONDITION_ORDER]
    controller = [payload["controller"][condition]["controller_accuracy"] for condition in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = ["#999999", "#4C78A8", "#E07B39"]
    for axis, values, title, ylabel in zip(
        axes,
        (selection, auroc, controller),
        ("Correct state selection", "Ranking improved vs harmed", "VLM-gated classifier"),
        ("Selection accuracy", "AUROC", "Classification accuracy"),
    ):
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=18)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(payload["controller"]["full_reasoning"]["always_after_accuracy"], color="#B22222", linestyle="--", label="Always adapted")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analyze(args: argparse.Namespace) -> None:
    records = collect_records(args.results_dir, args.output_dir)
    if not records:
        raise SystemExit("No complete reasoned-adaptation audit records found")
    available = sum(len(decisive_pairs(args.results_dir, method)) for method in METHOD_ORDER)
    payload = {
        "schema_version": 1,
        "vlm_model_id": "Qwen/Qwen3.6-35B-A3B",
        "thinking_enabled": False,
        "before_after_explicit": True,
        "continuous_adaptation": True,
        "episodic_reset": False,
        "ground_truth_visible_to_vlm": False,
        "conditions": {condition: condition_metrics(records, condition) for condition in CONDITION_ORDER},
        "controller": {condition: controller_metrics(args.results_dir, records, condition) for condition in CONDITION_ORDER},
        "comparisons": [
            paired_bootstrap(records, "full_reasoning", "image_predictions", args.bootstrap_resamples, args.seed),
            paired_bootstrap(records, "full_reasoning", "predictions_only", args.bootstrap_resamples, args.seed + 1),
        ],
        "coverage": {
            "available_decisive_pairs": available,
            "complete_decisive_pairs": len(records),
            "improved_pairs": sum(record["outcome"] == "improved" for record in records),
            "harmed_pairs": sum(record["outcome"] == "harmed" for record in records),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "reasoned_adaptation_metrics.json", payload)
    write_csv(args.output_dir / "reasoned_adaptation_records.csv", records)
    markdown = build_markdown(payload)
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    plot_results(args.output_dir / "reasoned_adaptation_results.png", payload)
    print(markdown)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "score":
        run_score(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
