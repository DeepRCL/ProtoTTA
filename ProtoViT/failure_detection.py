#!/usr/bin/env python3
"""Label-blind VLM failure detection and MSP comparison.

The scoring pass deliberately uses sanitized reasoning boards and never puts the
ground-truth class or correctness in the VLM prompt.  The analysis pass treats a
wrong classifier prediction as the positive class and compares:

* MSP failure score: 1 - maximum softmax probability
* VLM failure score: inverse of the label-blind 1--5 quality score
* Fixed combination: the untrained mean of the two failure scores
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


METHOD_ORDER = ["unadapted", "tent", "eata", "sar", "memo", "prototta"]
DISPLAY_NAMES = {
    "unadapted": "Unadapted",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "memo": "Memo",
    "prototta": "ProtoTTA",
}
PROMPT_NAME = "06_failure_detection_prompt.txt"
PREDICTED_BOARD_NAME = "06_blind_predicted_class_reasoning.png"
ANY_CLASS_BOARD_NAME = "06_blind_any_class_reasoning.png"
SCORE_NAME = "07_failure_detection.json"
RAW_NAME = "07_failure_detection_raw.txt"
ERROR_NAME = "07_failure_detection_error.txt"
LOGGER = logging.getLogger("failure_detection")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "vlm_eval",
        help="VLM artifact directory containing per-method sample folders.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Run label-blind VLM scoring.")
    score_parser.add_argument("--method", choices=METHOD_ORDER + ["all"], default="all")
    score_parser.add_argument(
        "--vlm-model-id", default="Qwen/Qwen3-VL-32B-Thinking", help="Hugging Face VLM id."
    )
    score_parser.add_argument("--max-new-tokens", type=int, default=2056)
    score_parser.add_argument("--max-samples", type=int, default=None)
    score_parser.add_argument("--overwrite", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Compute AUROC/AUPR and plots.")
    analyze_parser.add_argument("--method", choices=METHOD_ORDER + ["all"], default="all")
    analyze_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    analyze_parser.add_argument("--seed", type=int, default=42)
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RESULTS_DIR/failure_detection.",
    )
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


def method_sample_dirs(results_dir: Path, method: str) -> List[Path]:
    root = results_dir / method / "samples"
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "03_meta.json").exists())


def selected_methods(method: str) -> List[str]:
    return METHOD_ORDER if method == "all" else [method]


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def sanitize_board(source: Path, destination: Path, board_name: str) -> None:
    """Cover the label-bearing title while preserving all reasoning panels."""
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    cover_height = max(82, int(round(height * 0.10)))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, cover_height), fill="white")
    title_font = font(max(18, int(round(height * 0.024))))
    label = f"Label-blind {board_name}"
    box = draw.textbbox((0, 0), label, font=title_font)
    text_width = box[2] - box[0]
    draw.text(((width - text_width) / 2, max(5, cover_height * 0.20)), label, fill="black", font=title_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def build_prompt(predicted_class: str) -> str:
    return (
        "You are a label-free failure detector for a prototype-based bird classifier.\n"
        "The classifier predicts by matching input regions to learned prototype patches.\n\n"
        "You receive THREE IMAGES in this exact order:\n"
        "IMAGE 1: the raw corrupted test image.\n"
        "IMAGE 2: a label-blind predicted-class reasoning board. Panel A is the input; "
        "Panel B is the strongest-match heatmap; C1-C5 are retrieved training prototypes.\n"
        "IMAGE 3: a label-blind any-class reasoning board with the ten strongest prototypes "
        "across all classes; use it to identify conflicting or spurious evidence.\n\n"
        f"The classifier's predicted class is: {predicted_class}.\n"
        "The ground-truth label and prediction correctness are intentionally NOT provided. "
        "Do not assume the prediction is correct. Judge whether it should be trusted using only "
        "the visible bird, localization, prototype matches, and competing prototype evidence.\n\n"
        "Score each dimension from 1 to 5:\n"
        "REASONING_COHERENCE_SCORE: 1 means focus is background/noise or irrelevant anatomy; "
        "5 means focus is a clear, species-discriminative bird feature.\n"
        "PROTOTYPE_SUPPORT_SCORE: 1 means prototypes do not visually support the prediction; "
        "5 means they consistently and specifically support it.\n"
        "OVERALL_QUALITY_SCORE: 1 means the prediction is unsafe and likely wrong; "
        "5 means the reasoning is coherent and the prediction is highly trustworthy.\n\n"
        "Return exactly one JSON object with keys: reasoning_coherence_score, "
        "prototype_support_score, overall_quality_score, failure_rationale. "
        "The rationale must be one concise sentence. Return no other text."
    )


def validate_score(payload: Dict) -> Dict:
    cleaned = dict(payload)
    for key in ("reasoning_coherence_score", "prototype_support_score", "overall_quality_score"):
        value = float(payload[key])
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"{key} must be in [1, 5], received {value}")
        cleaned[key] = value
    if not str(payload.get("failure_rationale", "")).strip():
        raise ValueError("failure_rationale is missing")
    cleaned["failure_rationale"] = str(payload["failure_rationale"]).strip()
    return cleaned


def run_scoring(args: argparse.Namespace) -> None:
    # Importing here keeps the CPU analysis path independent of torch/transformers.
    from vlm_eval import VLMScorer

    methods = selected_methods(args.method)
    scorer = VLMScorer(args.vlm_model_id, args.max_new_tokens)
    processed = skipped = failed = 0

    for method in methods:
        sample_dirs = method_sample_dirs(args.results_dir, method)
        if args.max_samples is not None:
            sample_dirs = sample_dirs[: args.max_samples]
        for sample_dir in sample_dirs:
            output_path = sample_dir / SCORE_NAME
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue

            meta = read_json(sample_dir / "03_meta.json")
            source_predicted = sample_dir / "01_predicted_class_reasoning.png"
            source_any = sample_dir / "02_any_class_reasoning.png"
            raw_image = sample_dir / "00_corrupted_input.png"
            if not all(path.exists() for path in (raw_image, source_predicted, source_any)):
                LOGGER.warning("Skipping incomplete sample directory %s", sample_dir)
                failed += 1
                continue

            blind_predicted = sample_dir / PREDICTED_BOARD_NAME
            blind_any = sample_dir / ANY_CLASS_BOARD_NAME
            sanitize_board(source_predicted, blind_predicted, "predicted-class reasoning board")
            sanitize_board(source_any, blind_any, "any-class reasoning board")
            prompt = build_prompt(str(meta["predicted_class"]))
            (sample_dir / PROMPT_NAME).write_text(prompt, encoding="utf-8")

            try:
                parsed, raw_text, actual_prompt = scorer.generate_json(
                    [raw_image, blind_predicted, blind_any], prompt
                )
                scores = validate_score(parsed)
                (sample_dir / RAW_NAME).write_text(raw_text, encoding="utf-8")
                payload = {
                    "schema_version": 1,
                    "label_blind": True,
                    "method": method,
                    "sample_id": sample_dir.name,
                    "predicted_class": meta["predicted_class"],
                    "vlm_model_id": args.vlm_model_id,
                    "prompt_sha256": hashlib.sha256(actual_prompt.encode("utf-8")).hexdigest(),
                    **scores,
                    "raw_response": raw_text,
                }
                write_json(output_path, payload)
                (sample_dir / ERROR_NAME).unlink(missing_ok=True)
                processed += 1
                LOGGER.info("Scored %s/%s", method, sample_dir.name)
            except Exception as exc:
                failed += 1
                LOGGER.exception("Scoring failed for %s/%s", method, sample_dir.name)
                (sample_dir / ERROR_NAME).write_text(str(exc), encoding="utf-8")

    LOGGER.info("Label-blind scoring finished: processed=%d skipped=%d failed=%d", processed, skipped, failed)


def collect_records(results_dir: Path, methods: Sequence[str]) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    missing = {"score": 0, "msp": 0, "msp_prediction_mismatch": 0, "invalid_score": 0}
    for method in methods:
        for sample_dir in method_sample_dirs(results_dir, method):
            score_path = sample_dir / SCORE_NAME
            if not score_path.exists():
                missing["score"] += 1
                continue
            meta = read_json(sample_dir / "03_meta.json")
            if "max_softmax_probability" not in meta:
                missing["msp"] += 1
                continue
            if meta.get("msp_prediction_matches_saved") is False:
                missing["msp_prediction_mismatch"] += 1
                continue
            score = read_json(score_path)
            if score.get("label_blind") is not True:
                missing["invalid_score"] += 1
                continue
            quality = float(score["overall_quality_score"])
            msp = float(meta["max_softmax_probability"])
            if not (1.0 <= quality <= 5.0 and 0.0 <= msp <= 1.0):
                missing["invalid_score"] += 1
                continue
            vlm_failure = (5.0 - quality) / 4.0
            msp_failure = 1.0 - msp
            records.append(
                {
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "sample_id": sample_dir.name,
                    "sample_idx": int(meta["sample_idx"]),
                    "corruption_type": meta["corruption_type"],
                    "predicted_class": meta["predicted_class"],
                    "ground_truth_class": meta["ground_truth_class"],
                    "is_wrong": int(not bool(meta["is_correct"])),
                    "max_softmax_probability": msp,
                    "overall_quality_score": quality,
                    "msp_failure_score": msp_failure,
                    "vlm_failure_score": vlm_failure,
                    "combined_failure_score": 0.5 * (msp_failure + vlm_failure),
                    "failure_rationale": score["failure_rationale"],
                    "sample_dir": str(sample_dir),
                }
            )
    return records, missing


SCORE_COLUMNS = {
    "MSP": "msp_failure_score",
    "VLM quality": "vlm_failure_score",
    "MSP + VLM": "combined_failure_score",
}


def metric_pair(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def cluster_bootstrap(
    records: Sequence[Dict], resamples: int, seed: int
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Paired bootstrap by underlying sample, preserving method dependence."""
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["sample_id"], []).append(index)
    group_names = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    draws: Dict[str, Dict[str, List[float]]] = {
        name: {"auroc": [], "aupr": []} for name in SCORE_COLUMNS
    }
    labels_all = np.asarray([record["is_wrong"] for record in records], dtype=int)
    score_arrays = {
        name: np.asarray([record[column] for record in records], dtype=float)
        for name, column in SCORE_COLUMNS.items()
    }
    for _ in range(resamples):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.asarray([index for group in sampled_groups for index in groups[str(group)]], dtype=int)
        labels = labels_all[indices]
        if len(np.unique(labels)) < 2:
            continue
        for name, scores in score_arrays.items():
            auroc, aupr = metric_pair(labels, scores[indices])
            draws[name]["auroc"].append(auroc)
            draws[name]["aupr"].append(aupr)

    intervals: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name in SCORE_COLUMNS:
        intervals[name] = {}
        for metric in ("auroc", "aupr"):
            values = draws[name][metric]
            intervals[name][metric] = (
                float(np.quantile(values, 0.025)) if values else float("nan"),
                float(np.quantile(values, 0.975)) if values else float("nan"),
            )
    return intervals


def paired_delta_bootstrap(
    records: Sequence[Dict], resamples: int, seed: int
) -> Dict[str, Dict[str, List[float]]]:
    comparisons = {
        "VLM quality - MSP": ("VLM quality", "MSP"),
        "MSP + VLM - MSP": ("MSP + VLM", "MSP"),
        "MSP + VLM - VLM quality": ("MSP + VLM", "VLM quality"),
    }
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["sample_id"], []).append(index)
    group_names = np.asarray(sorted(groups))
    labels_all = np.asarray([record["is_wrong"] for record in records], dtype=int)
    score_arrays = {
        name: np.asarray([record[column] for record in records], dtype=float)
        for name, column in SCORE_COLUMNS.items()
    }
    draws = {comparison: {"auroc": [], "aupr": []} for comparison in comparisons}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.asarray([index for group in sampled_groups for index in groups[str(group)]], dtype=int)
        labels = labels_all[indices]
        if len(np.unique(labels)) < 2:
            continue
        sampled_metrics = {
            name: metric_pair(labels, scores[indices]) for name, scores in score_arrays.items()
        }
        for comparison, (left, right) in comparisons.items():
            draws[comparison]["auroc"].append(sampled_metrics[left][0] - sampled_metrics[right][0])
            draws[comparison]["aupr"].append(sampled_metrics[left][1] - sampled_metrics[right][1])

    output: Dict[str, Dict[str, List[float]]] = {}
    full_metrics = {
        name: metric_pair(labels_all, scores) for name, scores in score_arrays.items()
    }
    for comparison, (left, right) in comparisons.items():
        output[comparison] = {}
        for metric_index, metric in enumerate(("auroc", "aupr")):
            values = draws[comparison][metric]
            output[comparison][metric] = [
                float(full_metrics[left][metric_index] - full_metrics[right][metric_index]),
                float(np.quantile(values, 0.025)) if values else float("nan"),
                float(np.quantile(values, 0.975)) if values else float("nan"),
            ]
    return output


def summarize_group(records: Sequence[Dict], resamples: int, seed: int) -> Dict:
    labels = np.asarray([record["is_wrong"] for record in records], dtype=int)
    summary = {
        "n": len(records),
        "num_wrong": int(labels.sum()),
        "error_prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "metrics": {},
        "paired_differences": {},
    }
    if not records:
        return summary
    intervals = cluster_bootstrap(records, resamples, seed) if len(np.unique(labels)) == 2 else {}
    for name, column in SCORE_COLUMNS.items():
        scores = np.asarray([record[column] for record in records], dtype=float)
        auroc, aupr = metric_pair(labels, scores)
        summary["metrics"][name] = {
            "auroc": auroc,
            "aupr": aupr,
            "auroc_ci95": list(intervals.get(name, {}).get("auroc", (float("nan"), float("nan")))),
            "aupr_ci95": list(intervals.get(name, {}).get("aupr", (float("nan"), float("nan")))),
        }
    if len(np.unique(labels)) == 2:
        summary["paired_differences"] = paired_delta_bootstrap(records, resamples, seed)
    return summary


def format_metric(payload: Dict, key: str) -> str:
    value = payload[key]
    low, high = payload[f"{key}_ci95"]
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def build_markdown(summary: Dict, missing: Dict[str, int]) -> str:
    pooled = summary["pooled"]
    lines = [
        "# Label-free Failure Detection",
        "",
        "A wrong classifier prediction is the positive class. Higher detector scores mean greater failure risk.",
        "The VLM is scored with ground truth and correctness withheld, and the reasoning-board titles are sanitized.",
        "The combined detector is a fixed, label-free average: `0.5 * (1 - MSP) + 0.5 * ((5 - VLM quality) / 4)`.",
        "Confidence intervals are paired cluster-bootstrap 95% intervals over underlying samples.",
        "",
        "## Pooled result",
        "",
        f"- N: {pooled['n']} method-sample pairs",
        f"- Wrong predictions: {pooled['num_wrong']} ({100.0 * pooled['error_prevalence']:.1f}%)",
        "",
        "Detector | AUROC (95% CI) | AUPR (95% CI)",
        "-------- | -------------- | -------------",
    ]
    for name in SCORE_COLUMNS:
        payload = pooled["metrics"][name]
        lines.append(f"{name} | {format_metric(payload, 'auroc')} | {format_metric(payload, 'aupr')}")

    lines.extend(
        [
            "",
            "### Paired detector differences",
            "",
            "Positive values favor the detector named first. Intervals are paired by sample.",
            "",
            "Comparison | Delta AUROC (95% CI) | Delta AUPR (95% CI)",
            "---------- | --------------------- | --------------------",
        ]
    )
    for comparison, metrics in pooled["paired_differences"].items():
        auroc = metrics["auroc"]
        aupr = metrics["aupr"]
        lines.append(
            f"{comparison} | {auroc[0]:.3f} [{auroc[1]:.3f}, {auroc[2]:.3f}] | "
            f"{aupr[0]:.3f} [{aupr[1]:.3f}, {aupr[2]:.3f}]"
        )

    lines.extend(["", "## Per-method results", ""])
    for method in METHOD_ORDER:
        if method not in summary["by_method"]:
            continue
        item = summary["by_method"][method]
        lines.extend(
            [
                f"### {DISPLAY_NAMES[method]}",
                "",
                f"N={item['n']}; wrong={item['num_wrong']} ({100.0 * item['error_prevalence']:.1f}%).",
                "",
                "Detector | AUROC (95% CI) | AUPR (95% CI)",
                "-------- | -------------- | -------------",
            ]
        )
        for name in SCORE_COLUMNS:
            payload = item["metrics"][name]
            lines.append(f"{name} | {format_metric(payload, 'auroc')} | {format_metric(payload, 'aupr')}")
        lines.append("")

    lines.extend(
        [
            "## Coverage audit",
            "",
            f"- Missing label-blind VLM score: {missing['score']}",
            f"- Missing MSP: {missing['msp']}",
            f"- Excluded because reproduced prediction differed from saved prediction: {missing['msp_prediction_mismatch']}",
            f"- Invalid/non-blind scores: {missing['invalid_score']}",
            "",
            "The legacy `05_vlm.json` scores are intentionally excluded because their prompts disclosed the ground truth and correctness.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_records_csv(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_curves(path: Path, records: Sequence[Dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    labels = np.asarray([record["is_wrong"] for record in records], dtype=int)
    if len(np.unique(labels)) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"MSP": "#4C78A8", "VLM quality": "#F58518", "MSP + VLM": "#54A24B"}
    for name, column in SCORE_COLUMNS.items():
        scores = np.asarray([record[column] for record in records], dtype=float)
        auroc, aupr = metric_pair(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, label=f"{name} ({auroc:.3f})", color=colors[name], linewidth=2)
        axes[1].plot(recall, precision, label=f"{name} ({aupr:.3f})", color=colors[name], linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].axhline(labels.mean(), linestyle="--", color="gray", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Misclassification ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Misclassification precision-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> None:
    methods = selected_methods(args.method)
    records, missing = collect_records(args.results_dir, methods)
    if not records:
        raise SystemExit("No matched label-blind VLM scores and MSP values were found.")
    output_dir = args.output_dir or args.results_dir / "failure_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "positive_class": "wrong prediction",
        "score_direction": "higher means greater failure risk",
        "combination": "0.5 * (1 - MSP) + 0.5 * ((5 - VLM quality) / 4)",
        "bootstrap_unit": "sample_id",
        "bootstrap_resamples": args.bootstrap_resamples,
        "pooled": summarize_group(records, args.bootstrap_resamples, args.seed),
        "by_method": {},
        "coverage_audit": missing,
    }
    for offset, method in enumerate(methods):
        method_records = [record for record in records if record["method"] == method]
        if method_records:
            summary["by_method"][method] = summarize_group(
                method_records, args.bootstrap_resamples, args.seed + offset + 1
            )

    write_json(output_dir / "failure_detection_metrics.json", summary)
    write_records_csv(output_dir / "failure_detection_records.csv", records)
    (output_dir / "README.md").write_text(build_markdown(summary, missing), encoding="utf-8")
    plot_curves(output_dir / "failure_detection_curves.png", records)
    print(build_markdown(summary, missing))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    if args.command == "score":
        run_scoring(args)
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
