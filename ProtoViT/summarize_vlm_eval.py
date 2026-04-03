#!/usr/bin/env python3
"""CPU-only summary of VLM evaluation results from saved JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


METHOD_ORDER = ["unadapted", "memo", "sar", "tent", "eata", "prototta"]
DISPLAY_NAMES = {
    "unadapted": "Unadapted",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "memo": "Memo",
    "prototta": "ProtoTTA",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Summarize saved VLM evaluation JSONs on CPU.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "vlm_eval",
        help="Path to results/vlm_eval.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=script_dir / "results" / "vlm_eval" / "README_summary.md",
        help="Markdown summary output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_dirs(method_dir: Path) -> list[Path]:
    samples_dir = method_dir / "samples"
    if not samples_dir.exists():
        return []
    return sorted(path for path in samples_dir.iterdir() if path.is_dir())


def summarize_method(results_dir: Path, method: str, subset_size: int | None) -> dict:
    method_dir = results_dir / method
    scored = []
    correct_scores = []
    failed = []
    present = 0

    for sample_dir in sample_dirs(method_dir):
        meta_path = sample_dir / "03_meta.json"
        vlm_path = sample_dir / "05_vlm.json"
        error_path = sample_dir / "05_vlm_error.txt"

        if not meta_path.exists():
            continue
        present += 1
        meta = load_json(meta_path)
        if vlm_path.exists():
            vlm = load_json(vlm_path)
            overall = float(vlm["overall_adaptation_quality"])
            part = float(vlm["part_coherence_score"])
            proto = float(vlm["prototype_match_score"])
            scored.append({"overall": overall, "part": part, "proto": proto, "correct": bool(meta["is_correct"])})
            if meta["is_correct"]:
                correct_scores.append(overall)
        elif error_path.exists():
            failed.append({"sample": sample_dir.name, "error": error_path.read_text(encoding="utf-8").strip()})

    if scored:
        overall_scores = [item["overall"] for item in scored]
        part_scores = [item["part"] for item in scored]
        proto_scores = [item["proto"] for item in scored]
        payload = {
            "display_name": DISPLAY_NAMES[method],
            "num_present": present,
            "num_scored": len(scored),
            "num_failed": len(failed),
            "subset_size": subset_size,
            "coverage_pct": (100.0 * len(scored) / subset_size) if subset_size else None,
            "VAQ": mean(overall_scores),
            "VAQ_correct": mean(correct_scores) if correct_scores else None,
            "VAQ_std": pstdev(overall_scores),
            "part_coherence_mean": mean(part_scores),
            "part_coherence_std": pstdev(part_scores),
            "prototype_match_mean": mean(proto_scores),
            "prototype_match_std": pstdev(proto_scores),
            "failed_samples": failed,
        }
    else:
        payload = {
            "display_name": DISPLAY_NAMES[method],
            "num_present": present,
            "num_scored": 0,
            "num_failed": len(failed),
            "subset_size": subset_size,
            "coverage_pct": 0.0 if subset_size else None,
            "VAQ": None,
            "VAQ_correct": None,
            "VAQ_std": None,
            "part_coherence_mean": None,
            "part_coherence_std": None,
            "prototype_match_mean": None,
            "prototype_match_std": None,
            "failed_samples": failed,
        }
    return payload


def format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def build_summary_table(summary: dict[str, dict]) -> str:
    lines = [
        "Method        | N Scored | N Failed | VAQ ↑ | VAQ_correct ↑ | VAQ Std | Part Coh. ↑ | Part Std | Proto Match ↑ | Proto Std",
        "------------- | -------- | -------- | ----- | ------------- | ------- | ----------- | -------- | ------------- | ---------",
    ]
    for method in METHOD_ORDER:
        item = summary[method]
        lines.append(
            f"{item['display_name']:<13} | "
            f"{item['num_scored']:<8} | "
            f"{item['num_failed']:<8} | "
            f"{format_value(item['VAQ']):<5} | "
            f"{format_value(item['VAQ_correct']):<13} | "
            f"{format_value(item['VAQ_std']):<7} | "
            f"{format_value(item['part_coherence_mean']):<11} | "
            f"{format_value(item['part_coherence_std']):<8} | "
            f"{format_value(item['prototype_match_mean']):<13} | "
            f"{format_value(item['prototype_match_std'])}"
        )
    return "\n".join(lines)


def build_markdown(results_dir: Path, summary: dict[str, dict], subset_size: int | None) -> str:
    lines = [
        "# VLM Evaluation Summary",
        "",
        f"- Results directory: `{results_dir}`",
        f"- Fixed subset size: `{subset_size}`" if subset_size is not None else "- Fixed subset size: unknown",
        "",
        "## Table",
        "",
        build_summary_table(summary),
        "",
        "## Metric Meanings",
        "",
        "- `VAQ`: mean overall adaptation quality score from the VLM. Higher means the VLM judged the model's reasoning to be more semantically convincing overall.",
        "- `VAQ_std`: standard deviation of `VAQ` across scored samples. Higher means the method's reasoning quality is less consistent across examples.",
        "- `VAQ_correct`: mean `VAQ` restricted to samples where the method predicted the correct class.",
        "- `Part Coh.`: mean part coherence score. Higher means the model focused on more meaningful or discriminative bird parts instead of background/noise.",
        "- `Part Std`: standard deviation of the part coherence score across scored samples.",
        "- `Proto Match`: mean prototype match score. Higher means the retrieved prototype patches looked more visually similar to the highlighted image region.",
        "- `Proto Std`: standard deviation of the prototype match score across scored samples.",
        "- `N Scored`: number of subset samples with a valid parsed VLM JSON.",
        "- `N Failed`: number of subset samples where the VLM output failed parsing or generation.",
        "",
        "## Reporting Guidance",
        "",
        "- Use methods with high coverage for the main paper table. `97/100` or `98/100` is reasonable to report.",
        "- Partial methods can be reported only if you explicitly disclose the sample count.",
        "- A method with very low `N Scored` should be treated as preliminary, not directly comparable to nearly complete runs.",
        "",
        "## Method Notes",
        "",
    ]

    for method in METHOD_ORDER:
        item = summary[method]
        lines.append(f"### {item['display_name']}")
        lines.append("")
        lines.append(f"- `N Scored`: {item['num_scored']}")
        lines.append(f"- `N Failed`: {item['num_failed']}")
        if item["coverage_pct"] is not None:
            lines.append(f"- `Coverage`: {item['coverage_pct']:.1f}% of the 100-sample subset")
        lines.append(f"- `VAQ`: {format_value(item['VAQ'])}")
        lines.append(f"- `VAQ_correct`: {format_value(item['VAQ_correct'])}")
        lines.append(f"- `VAQ_std`: {format_value(item['VAQ_std'])}")
        lines.append(f"- `Part Coh.`: {format_value(item['part_coherence_mean'])}")
        lines.append(f"- `Part Std`: {format_value(item['part_coherence_std'])}")
        lines.append(f"- `Proto Match`: {format_value(item['prototype_match_mean'])}")
        lines.append(f"- `Proto Std`: {format_value(item['prototype_match_std'])}")
        if item["num_scored"] == 0:
            lines.append("- Status: not usable yet for the paper.")
        elif subset_size and item["num_scored"] < subset_size * 0.8:
            lines.append("- Status: partial result only; include only with a clear sample-count disclaimer.")
        else:
            lines.append("- Status: usable for the paper with sample count reported.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_md = args.output_md.resolve()

    subset_path = results_dir / "subset.json"
    subset_size = None
    if subset_path.exists():
        subset = load_json(subset_path)
        subset_size = len(subset.get("samples", []))

    summary = {method: summarize_method(results_dir, method, subset_size) for method in METHOD_ORDER}
    table = build_summary_table(summary)
    markdown = build_markdown(results_dir, summary, subset_size)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    payload = {
        "results_dir": str(results_dir),
        "subset_size": subset_size,
        "methods": summary,
        "summary_table": table,
    }
    with (results_dir / "cpu_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(table)
    print(f"\nWrote markdown summary to {output_md}")


if __name__ == "__main__":
    main()
