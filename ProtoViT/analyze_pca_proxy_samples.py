#!/usr/bin/env python3
"""CPU-only analysis of PCA-W-like prototype purity proxy vs saved VLM scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


METHOD_ORDER = ["unadapted", "tent", "eata", "sar", "memo", "prototta"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze PCA-W proxy from saved VLM sample metadata.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "vlm_eval",
        help="Path to results/vlm_eval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many high/low examples to report.",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_ORDER + ["all"],
        default="all",
        help="Restrict analysis to one method or use all available methods.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=script_dir / "results" / "vlm_eval" / "PCAW_PROXY_ANALYSIS.md",
        help="Markdown output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_dirs(results_dir: Path, method: str) -> List[Path]:
    sample_root = results_dir / method / "samples"
    if not sample_root.exists():
        return []
    return sorted(path for path in sample_root.iterdir() if path.is_dir())


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    return pearson(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def compute_proxy(meta: Dict) -> Dict[str, float]:
    gt_idx = int(meta["ground_truth_index"])
    any_top = meta.get("any_class_top_prototypes", [])
    pred_top = meta.get("predicted_top_prototypes", [])

    any_contribs = [max(float(item.get("contribution", 0.0)), 0.0) for item in any_top]
    any_total = float(sum(any_contribs))
    any_gt = float(
        sum(
            max(float(item.get("contribution", 0.0)), 0.0)
            for item in any_top
            if int(item.get("class_index", item.get("class", -1))) == gt_idx
        )
    )
    pred_contribs = [max(float(item.get("contribution", 0.0)), 0.0) for item in pred_top]
    pred_total = float(sum(pred_contribs))
    pred_gt = float(
        sum(
            max(float(item.get("contribution", 0.0)), 0.0)
            for item in pred_top
            if int(item.get("class_index", item.get("class", -1))) == gt_idx
        )
    )

    any_wrong_count = sum(
        1 for item in any_top if int(item.get("class_index", item.get("class", -1))) != gt_idx
    )
    top1_is_gt = (
        int(any_top[0].get("class_index", any_top[0].get("class", -1))) == gt_idx if any_top else False
    )

    return {
        "pcaw_proxy_any_ratio": (any_gt / any_total) if any_total > 0 else 0.0,
        "pcaw_proxy_pred_ratio": (pred_gt / pred_total) if pred_total > 0 else 0.0,
        "gt_any_contribution_sum": any_gt,
        "any_contribution_sum": any_total,
        "any_wrong_class_count": int(any_wrong_count),
        "top1_any_is_gt": bool(top1_is_gt),
    }


def build_records(results_dir: Path, methods: List[str]) -> List[Dict]:
    records: List[Dict] = []
    for method in methods:
        for sample_dir in sample_dirs(results_dir, method):
            meta_path = sample_dir / "03_meta.json"
            if not meta_path.exists():
                continue
            meta = load_json(meta_path)
            proxy = compute_proxy(meta)
            record = {
                "method": method,
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "raw_image_path": meta.get("raw_image_path"),
                "figure_path": meta.get("figure_path"),
                "any_class_figure_path": meta.get("any_class_figure_path"),
                "ground_truth_class": meta.get("ground_truth_class"),
                "predicted_class": meta.get("predicted_class"),
                "is_correct": bool(meta.get("is_correct")),
                **proxy,
            }

            vlm_path = sample_dir / "05_vlm.json"
            if vlm_path.exists():
                vlm = load_json(vlm_path)
                record["vlm_overall"] = float(vlm["overall_adaptation_quality"])
                record["vlm_part"] = float(vlm["part_coherence_score"])
                record["vlm_proto"] = float(vlm["prototype_match_score"])
            else:
                record["vlm_overall"] = None
                record["vlm_part"] = None
                record["vlm_proto"] = None
            records.append(record)
    return records


def correlation_block(records: List[Dict]) -> Dict[str, Optional[float]]:
    valid = [r for r in records if r["vlm_overall"] is not None]
    x = [r["pcaw_proxy_any_ratio"] for r in valid]
    return {
        "n": len(valid),
        "pearson_overall": pearson(x, [r["vlm_overall"] for r in valid]),
        "spearman_overall": spearman(x, [r["vlm_overall"] for r in valid]),
        "pearson_part": pearson(x, [r["vlm_part"] for r in valid]),
        "spearman_part": spearman(x, [r["vlm_part"] for r in valid]),
        "pearson_proto": pearson(x, [r["vlm_proto"] for r in valid]),
        "spearman_proto": spearman(x, [r["vlm_proto"] for r in valid]),
    }


def format_num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_record(record: Dict) -> str:
    return (
        f"- `{record['method']}` / `{record['sample_id']}` | proxy={record['pcaw_proxy_any_ratio']:.3f} | "
        f"VLM={record['vlm_overall'] if record['vlm_overall'] is not None else 'n/a'} | "
        f"GT={record['ground_truth_class']} | Pred={record['predicted_class']} | "
        f"[dir]({record['sample_dir']})"
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    methods = METHOD_ORDER if args.method == "all" else [args.method]
    records = build_records(results_dir, methods)
    if not records:
        raise SystemExit("No sample metadata found.")

    high = sorted(records, key=lambda item: item["pcaw_proxy_any_ratio"], reverse=True)[: args.top_k]
    low = sorted(records, key=lambda item: item["pcaw_proxy_any_ratio"])[: args.top_k]
    corr = correlation_block(records)

    lines = [
        "# PCA-W Proxy Analysis",
        "",
        "This file uses a CPU-only proxy, not the exact PCA-W metric.",
        "",
        "Proxy definition:",
        "",
        "`pcaw_proxy_any_ratio = sum(correct-class contributions in any_class_top_prototypes) / sum(all contributions in any_class_top_prototypes)`",
        "",
        "Interpretation:",
        "- Higher values mean a larger fraction of the strongest contributing prototypes come from the ground-truth class.",
        "- Lower values mean more contribution mass comes from wrong-class or spurious prototypes.",
        "",
        "## Correlation With Saved VLM Scores",
        "",
        f"- N with VLM scores: {corr['n']}",
        f"- Pearson(proxy, overall_adaptation_quality): {format_num(corr['pearson_overall'])}",
        f"- Spearman(proxy, overall_adaptation_quality): {format_num(corr['spearman_overall'])}",
        f"- Pearson(proxy, part_coherence_score): {format_num(corr['pearson_part'])}",
        f"- Spearman(proxy, part_coherence_score): {format_num(corr['spearman_part'])}",
        f"- Pearson(proxy, prototype_match_score): {format_num(corr['pearson_proto'])}",
        f"- Spearman(proxy, prototype_match_score): {format_num(corr['spearman_proto'])}",
        "",
        f"## Top {args.top_k} Highest Proxy Samples",
        "",
    ]
    lines.extend(format_record(record) for record in high)
    lines.extend(["", f"## Top {args.top_k} Lowest Proxy Samples", ""])
    lines.extend(format_record(record) for record in low)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "methods": methods,
        "correlation": corr,
        "top_high": high,
        "top_low": low,
    }
    with (results_dir / "pcaw_proxy_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n".join(lines[:20]))
    print(f"\nWrote analysis to {args.output_md}")


if __name__ == "__main__":
    main()
