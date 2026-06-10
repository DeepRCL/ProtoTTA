#!/usr/bin/env python3
"""Generate Markdown and LaTeX comparison tables from Cars-C robustness JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CORRUPTION_GROUPS = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "Blur": ["defocus_blur", "gaussian_blur", "motion_blur"],
    "Weather": ["brightness", "fog", "frost", "spatter"],
    "Digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
}

METHOD_LABELS = {
    "normal": "Unadapted",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "proto_tta": "ProtoTTA",
    "proto_tta_plus": "ProtoTTA+",
}

DEFAULT_METHOD_ORDER = [
    "normal",
    "tent",
    "eata",
    "sar",
    "proto_tta",
    "proto_tta_plus",
]


def load_results(path: Path):
    obj = json.loads(path.read_text())
    return obj.get("metadata", {}), obj.get("results", {}), obj.get("aggregates", {})


def collect_eval_configs(results, methods, severity):
    configs = []
    for method in methods:
        for corruption_map in results.get(method, {}).values():
            entry = corruption_map.get(str(severity))
            if isinstance(entry, dict):
                cfg = entry.get("eval_config")
                if isinstance(cfg, dict):
                    configs.append(cfg)
    return configs


def normalize_shared_eval_config(cfg):
    if not isinstance(cfg, dict):
        return {}
    ignored = {"method", "method_config"}
    return {k: v for k, v in cfg.items() if k not in ignored}


def first_eval_config(results, methods, severity):
    for method in methods:
        for corruption_map in results.get(method, {}).values():
            entry = corruption_map.get(str(severity))
            if isinstance(entry, dict):
                cfg = entry.get("eval_config")
                if isinstance(cfg, dict):
                    return cfg
    return {}


def get_entry(results, method, corruption, severity):
    entry = results.get(method, {}).get(corruption, {}).get(str(severity))
    return entry if isinstance(entry, dict) else None


def get_metric(results, method, corruption, severity, metric):
    entry = get_entry(results, method, corruption, severity)
    if not entry:
        return None
    if metric == "relative_speed":
        baseline = get_entry(results, "normal", corruption, severity)
        if not baseline:
            return None
        base_t = baseline.get("efficiency", {}).get("time_per_sample_ms")
        curr_t = entry.get("efficiency", {}).get("time_per_sample_ms")
        if base_t is None or curr_t is None:
            return None
        return base_t / max(curr_t, 1e-8)
    if metric == "time_per_sample_ms":
        return entry.get("efficiency", {}).get("time_per_sample_ms")
    return entry.get(metric)


def available_methods(results, severity):
    methods = []
    for method, corruption_dict in results.items():
        has_value = any(
            isinstance(corruption_dict.get(c, {}).get(str(severity)), dict)
            for c in corruption_dict
        )
        if has_value:
            methods.append(method)
    return methods


def present_corruptions(results, methods, severity):
    corrs = []
    seen = set()
    for method in methods:
        for corruption in results.get(method, {}):
            if get_entry(results, method, corruption, severity) and corruption not in seen:
                corrs.append(corruption)
                seen.add(corruption)
    return corrs


def summarize_by_method(results, methods, corruptions, severity, metric):
    summary = {}
    for method in methods:
        values = [get_metric(results, method, corruption, severity, metric) for corruption in corruptions]
        values = [v for v in values if v is not None]
        if values:
            summary[method] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def rank_methods(cell_values, reverse=True):
    valid = [(m, v) for m, v in cell_values.items() if v is not None]
    if len(valid) < 2:
        return {m: "" for m in cell_values}
    ordered = sorted(valid, key=lambda x: x[1], reverse=reverse)
    best = ordered[0][0]
    second = ordered[1][0]
    marks = {m: "" for m in cell_values}
    marks[best] = "best"
    marks[second] = "second"
    return marks


def fmt_pct(value):
    return "N/A" if value is None else f"{value * 100:.1f}"


def fmt_pct_with_std(mean, std):
    return f"{mean * 100:.1f} ± {std * 100:.1f}"


def emphasize_md(text, mark):
    if mark == "best":
        return f"**{text}**"
    if mark == "second":
        return f"*{text}*"
    return text


def build_accuracy_table(results, methods, severity):
    available = set(present_corruptions(results, methods, severity))
    group_columns = {
        group: [c for c in names if c in available]
        for group, names in CORRUPTION_GROUPS.items()
    }

    rows = []
    for method in methods:
        row = {}
        flat_corrs = []
        for group, names in group_columns.items():
            for corruption in names:
                row[corruption] = get_metric(results, method, corruption, severity, "accuracy")
                flat_corrs.append(corruption)
            vals = [row[n] for n in names if row[n] is not None]
            row[f"{group}_avg"] = statistics.mean(vals) if vals else None
        total_vals = [row[c] for c in flat_corrs if row[c] is not None]
        row["Total"] = statistics.mean(total_vals) if total_vals else None
        row["Total_std"] = statistics.pstdev(total_vals) if len(total_vals) > 1 else 0.0
        rows.append((method, row))

    columns = []
    for group, names in group_columns.items():
        columns.extend(names)
        if names:
            columns.append(f"{group} Avg")
            columns.append(f"{group}_avg")
    columns.append("Total")

    ranks = {}
    for col in columns:
        ranks[col] = rank_methods({m: row[col] for m, row in rows if col in row}, reverse=True)

    return rows, ranks, group_columns


def build_metrics_table(results, methods, severity):
    corruptions = present_corruptions(results, methods, severity)
    pac = summarize_by_method(results, methods, corruptions, severity, "PAC_mean")
    pca_w = summarize_by_method(results, methods, corruptions, severity, "PCA_weighted_mean")
    stability = summarize_by_method(results, methods, corruptions, severity, "prediction_stability")
    selection = summarize_by_method(results, methods, corruptions, severity, "selection_rate")
    rel_speed = summarize_by_method(results, methods, corruptions, severity, "relative_speed")
    return pac, pca_w, stability, selection, rel_speed


def build_metric_ranks(methods, pac, pca_w, stability, selection, rel_speed):
    return {
        "PAC": rank_methods({m: pac.get(m, {}).get("mean") for m in methods}, reverse=True),
        "PCA-W": rank_methods({m: pca_w.get(m, {}).get("mean") for m in methods}, reverse=True),
        "Stability": rank_methods({m: stability.get(m, {}).get("mean") for m in methods}, reverse=True),
        "Selection": rank_methods({m: selection.get(m, {}).get("mean") for m in methods}, reverse=False),
        "RelSpeed": rank_methods({m: rel_speed.get(m, {}).get("mean") for m in methods}, reverse=True),
    }


def generate_markdown(metadata, results, methods, severity):
    rows, ranks, group_columns = build_accuracy_table(results, methods, severity)
    pac, pca_w, stability, selection, rel_speed = build_metrics_table(results, methods, severity)
    metric_ranks = build_metric_ranks(methods, pac, pca_w, stability, selection, rel_speed)
    cfg = first_eval_config(results, methods, severity)

    md = []
    md.append("# Cars-C Robustness Report\n")
    md.append(f"- Checkpoint: `{metadata.get('ckpt', 'N/A')}`")
    md.append(f"- Severity: `{severity}`")
    md.append(f"- Baseline adapt mode: `{metadata.get('baseline_adapt_mode', cfg.get('baseline_adapt_mode', 'N/A'))}`")
    md.append(f"- Proto adapt mode: `{metadata.get('proto_adapt_mode', cfg.get('proto_adapt_mode', 'N/A'))}`")
    md.append(f"- Batch size: `{metadata.get('batch_size', cfg.get('batch_size', 'N/A'))}`")
    md.append(f"- LR / Proto LR: `{metadata.get('lr', cfg.get('lr', 'N/A'))}` / `{metadata.get('proto_lr', cfg.get('proto_lr', 'N/A'))}`")
    md.append(f"- Proto threshold: `{metadata.get('proto_threshold', cfg.get('proto_threshold', 'N/A'))}`")
    md.append(f"- Proto conf threshold: `{metadata.get('proto_conf_threshold', cfg.get('proto_conf_threshold', 'N/A'))}`")
    md.append(f"- Proto rel threshold: `{metadata.get('proto_target_rel_threshold', cfg.get('proto_target_rel_threshold', 'N/A'))}`")
    md.append(f"- Methods: {', '.join(METHOD_LABELS.get(m, m) for m in methods)}\n")

    headers = ["Method"]
    order = []
    short = {
        "gaussian_noise": "Gauss",
        "shot_noise": "Shot",
        "impulse_noise": "Impul",
        "speckle_noise": "Speck",
        "defocus_blur": "Defoc",
        "gaussian_blur": "GBlur",
        "motion_blur": "MBlur",
        "brightness": "Bright",
        "fog": "Fog",
        "frost": "Frost",
        "spatter": "Spatt",
        "contrast": "Contr",
        "elastic_transform": "Elast",
        "jpeg_compression": "Jpeg",
        "pixelate": "Pixel",
    }
    for group, names in group_columns.items():
        for name in names:
            headers.append(short.get(name, name))
            order.append(name)
        if names:
            headers.append(f"{group} Avg")
            order.append(f"{group}_avg")
    headers.append("Total")
    order.append("Total")

    md.append("## Accuracy Comparison\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for method, row in rows:
        vals = []
        for col in order:
            if col == "Total":
                txt = f"{row['Total']*100:.1f} ± {row['Total_std']*100:.1f}" if row.get("Total") is not None else "N/A"
            else:
                txt = fmt_pct(row.get(col))
            vals.append(emphasize_md(txt, ranks[col].get(method, "")))
        md.append("| " + " | ".join([METHOD_LABELS.get(method, method)] + vals) + " |")

    md.append("\n## Efficiency and Interpretability\n")
    md.append("| Method | PAC | PCA-W | Prediction Stability | Selection Rate | Rel. Speed |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        pac_txt = fmt_pct_with_std(pac[method]["mean"], pac[method]["std"]) if method in pac else "N/A"
        pca_txt = fmt_pct_with_std(pca_w[method]["mean"], pca_w[method]["std"]) if method in pca_w else "N/A"
        stab_txt = fmt_pct_with_std(stability[method]["mean"], stability[method]["std"]) if method in stability else "N/A"
        sel_txt = f"{selection[method]['mean']*100:.1f}%" if method in selection else "N/A"
        speed_txt = f"{rel_speed[method]['mean']*100:.1f}%" if method in rel_speed else "N/A"

        pac_txt = emphasize_md(pac_txt, metric_ranks["PAC"].get(method, ""))
        pca_txt = emphasize_md(pca_txt, metric_ranks["PCA-W"].get(method, ""))
        stab_txt = emphasize_md(stab_txt, metric_ranks["Stability"].get(method, ""))
        sel_txt = emphasize_md(sel_txt, metric_ranks["Selection"].get(method, ""))
        speed_txt = emphasize_md(speed_txt, metric_ranks["RelSpeed"].get(method, ""))
        md.append(f"| {METHOD_LABELS.get(method, method)} | {pac_txt} | {pca_txt} | {stab_txt} | {sel_txt} | {speed_txt} |")

    return "\n".join(md) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown reports from Cars-C robustness JSON.")
    parser.add_argument("--input", default="results/cars_c_v5.json", help="Input JSON path")
    parser.add_argument("--severity", type=int, default=5, help="Severity key to report")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for .md files")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    metadata, results, _ = load_results(input_path)
    methods = [m for m in DEFAULT_METHOD_ORDER if m in available_methods(results, args.severity)]
    configs = collect_eval_configs(results, methods, args.severity)
    
    if configs:
        canonical = json.dumps(normalize_shared_eval_config(configs[0]), sort_keys=True)
        for cfg in configs[1:]:
            if json.dumps(normalize_shared_eval_config(cfg), sort_keys=True) != canonical:
                print(f"Warning: Mixed eval_config entries found in {input_path}.")
    
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_suffix("")
    md = generate_markdown(metadata, results, methods, args.severity)

    md_path = output_prefix.with_name(output_prefix.name + "_report.md")
    md_path.write_text(md)

    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
