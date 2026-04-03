#!/usr/bin/env python3
"""Generate Markdown and LaTeX comparison tables from Dogs robustness JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


CORRUPTION_GROUPS = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "Blur": ["defocus_blur", "gaussian_blur"],
    "Weather": ["brightness", "fog", "frost"],
    "Digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
}

METHOD_LABELS = {
    "normal": "Unadapted",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "proto_tta": "ProtoTTA",
    "proto_tta_plus_7030": "ProtoTTA+ (0.7/0.3)",
    "proto_tta_plus_7525": "ProtoTTA+ (0.75/0.25)",
    "proto_tta_plus_8020": "ProtoTTA+ (0.8/0.2)",
}


def load_results(path: Path):
    obj = json.loads(path.read_text())
    return obj.get("metadata", {}), obj["results"]


def get_entry(results, method, corruption, severity):
    entry = results.get(method, {}).get(corruption, {}).get(str(severity))
    return entry if isinstance(entry, dict) else None


def get_metric(results, method, corruption, severity, metric):
    entry = get_entry(results, method, corruption, severity)
    if not entry:
        return None
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


def summarize_by_method(results, methods, corruptions, severity, metric):
    summary = {}
    for method in methods:
        values = [
            get_metric(results, method, corruption, severity, metric)
            for corruption in corruptions
        ]
        values = [v for v in values if v is not None]
        if values:
            summary[method] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def rank_methods(cell_values):
    valid = [(m, v) for m, v in cell_values.items() if v is not None]
    if len(valid) < 2:
        return {m: "" for m in cell_values}
    ordered = sorted(valid, key=lambda x: x[1], reverse=True)
    best = ordered[0][0]
    second = ordered[1][0]
    marks = {m: "" for m in cell_values}
    marks[best] = "best"
    marks[second] = "second"
    return marks


def fmt_pct_raw(value):
    return "N/A" if value is None else f"{value * 100:.1f}"


def fmt_pct(value):
    return "N/A" if value is None else f"{value * 100:.1f}"


def fmt_mean_std(mean, std):
    return f"{mean * 100:.1f} $\\pm$ {std * 100:.1f}"


def emphasize_latex(text, mark):
    if mark == "best":
        return f"\\textbf{{{text}}}"
    if mark == "second":
        return f"\\underline{{{text}}}"
    return text


def emphasize_md(text, mark):
    if mark == "best":
        return f"**{text}**"
    if mark == "second":
        return f"*{text}*"
    return text


def build_accuracy_table(results, methods, severity):
    columns = []
    for group, names in CORRUPTION_GROUPS.items():
        columns.extend(names)
        columns.append(f"{group}_avg")
    columns.append("Total")

    rows = []
    for method in methods:
        row = {}
        for corruption in [c for names in CORRUPTION_GROUPS.values() for c in names]:
            row[corruption] = get_metric(results, method, corruption, severity, "accuracy")
        for group, names in CORRUPTION_GROUPS.items():
            vals = [row[n] for n in names if row[n] is not None]
            row[f"{group}_avg"] = statistics.mean(vals) if vals else None
        total_vals = [row[n] for n in [c for names in CORRUPTION_GROUPS.values() for c in names] if row[n] is not None]
        row["Total"] = statistics.mean(total_vals) if total_vals else None
        row["Total_std"] = statistics.pstdev(total_vals) if len(total_vals) > 1 else 0.0
        rows.append((method, row))

    ranks = {col: rank_methods({m: row[col] for m, row in rows}) for col in columns if col != "Total"}
    ranks["Total"] = rank_methods({m: row["Total"] for m, row in rows})
    return rows, ranks


def build_efficiency_table(results, methods, severity):
    corruptions = [c for names in CORRUPTION_GROUPS.values() for c in names]
    pac = summarize_by_method(results, methods, corruptions, severity, "PAC_mean")
    pca_w = summarize_by_method(results, methods, corruptions, severity, "PCA_weighted_mean")
    calib = summarize_by_method(results, methods, corruptions, severity, "calibration_agreement")
    adapt = summarize_by_method(results, methods, corruptions, severity, "adaptation_rate")
    speed = {}

    base_speeds = [
        get_entry(results, "normal", c, severity).get("efficiency", {}).get("time_per_sample_ms")
        for c in corruptions
        if get_entry(results, "normal", c, severity)
    ]
    base_speed = statistics.mean([v for v in base_speeds if v is not None]) if base_speeds else None

    for method in methods:
        speeds = []
        for corruption in corruptions:
            entry = get_entry(results, method, corruption, severity)
            if entry and "efficiency" in entry:
                val = entry["efficiency"].get("time_per_sample_ms")
                if val is not None:
                    speeds.append(val)
        if speeds:
            mean_speed = statistics.mean(speeds)
            rel = (base_speed / mean_speed) * 100 if base_speed and mean_speed else None
            speed[method] = {"mean": mean_speed, "relative": rel}

    return pac, pca_w, calib, adapt, speed


def generate_markdown(metadata, results, methods, severity):
    rows, ranks = build_accuracy_table(results, methods, severity)
    pac, pca_w, calib, adapt, speed = build_efficiency_table(results, methods, severity)

    flat_corrs = [c for names in CORRUPTION_GROUPS.values() for c in names]
    md = []
    md.append(f"# Dogs Robustness Report\n")
    md.append(f"- Model: `{metadata.get('model', 'N/A')}`")
    md.append(f"- Severity: `{severity}`")
    md.append(f"- Methods: {', '.join(METHOD_LABELS.get(m, m) for m in methods)}\n")

    headers = ["Method", "Gauss", "Shot", "Impul", "Speck", "Noise Avg", "Defoc", "GBlur", "Blur Avg",
               "Brit", "Fog", "Frost", "Weather Avg", "Contr", "Elast", "Jpeg", "Pixel", "Digital Avg", "Total"]
    md.append("## Accuracy Comparison\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for method, row in rows:
        vals = []
        order = [
            "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "Noise_avg",
            "defocus_blur", "gaussian_blur", "Blur_avg",
            "brightness", "fog", "frost", "Weather_avg",
            "contrast", "elastic_transform", "jpeg_compression", "pixelate", "Digital_avg",
            "Total",
        ]
        for col in order:
            if col.endswith("_avg") or col == "Total":
                key = col if col == "Total" else col
            else:
                key = col
            val = row[key]
            text = fmt_pct(val) if col != "Total" else f"{row['Total']*100:.1f} ± {row['Total_std']*100:.1f}"
            vals.append(emphasize_md(text, ranks["Total" if col == "Total" else key].get(method, "")))
        md.append("| " + " | ".join([METHOD_LABELS.get(method, method)] + vals) + " |")

    md.append("\n## Efficiency and Interpretability\n")
    md.append("| Method | PAC | PCA-W | Prediction Stability | Selection Rate | Rel. Speed |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        pac_txt = fmt_mean_std(pac[method]["mean"], pac[method]["std"]) if method in pac else "N/A"
        pca_txt = fmt_mean_std(pca_w[method]["mean"], pca_w[method]["std"]) if method in pca_w else "N/A"
        calib_txt = fmt_mean_std(calib[method]["mean"], calib[method]["std"]) if method in calib else "N/A"
        adapt_txt = f"{adapt[method]['mean']*100:.1f}%" if method in adapt else "N/A"
        speed_txt = f"{speed[method]['relative']:.1f}%" if method in speed and speed[method]["relative"] is not None else "N/A"
        md.append(f"| {METHOD_LABELS.get(method, method)} | {pac_txt} | {pca_txt} | {calib_txt} | {adapt_txt} | {speed_txt} |")

    return "\n".join(md) + "\n"


def generate_latex(metadata, results, methods, severity):
    rows, ranks = build_accuracy_table(results, methods, severity)
    pac, pca_w, calib, adapt, speed = build_efficiency_table(results, methods, severity)

    tex = []
    tex.append("\\begin{table*}[t!]")
    tex.append("\\centering")
    tex.append("\\caption{Test-time adaptation performance on Stanford Dogs-C across corruption types. Best results in \\textbf{bold}, second-best \\underline{underlined}.}")
    tex.append("\\label{tab:dogs_results}")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append("\\begin{tabular}{l|cccc>{\\columncolor{gray!5}}c|cc>{\\columncolor{gray!5}}c|ccc>{\\columncolor{gray!5}}c|cccc>{\\columncolor{gray!5}}c|c}")
    tex.append("\\toprule")
    tex.append("& \\multicolumn{5}{c|}{\\textbf{Noise}} & \\multicolumn{3}{c|}{\\textbf{Blur}} & \\multicolumn{4}{c|}{\\textbf{Weather}} & \\multicolumn{5}{c|}{\\textbf{Digital}} & \\\\")
    tex.append("\\textbf{Method} & Gauss & Shot & Impul & Speck & \\textbf{Avg} & Defoc & Gauss & \\textbf{Avg} & Brit & Fog & Frost & \\textbf{Avg} & Contr & Elast & Jpeg & Pixel & \\textbf{Avg} & \\textbf{Total} \\\\")
    tex.append("\\midrule")
    tex.append("\\multicolumn{19}{c}{\\textit{\\textbf{Backbone: ProtoPFormer ~|~ Dataset: Stanford Dogs-C}}} \\\\")
    tex.append("\\midrule")

    order = [
        "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "Noise_avg",
        "defocus_blur", "gaussian_blur", "Blur_avg",
        "brightness", "fog", "frost", "Weather_avg",
        "contrast", "elastic_transform", "jpeg_compression", "pixelate", "Digital_avg",
    ]

    for method, row in rows:
        cells = []
        for col in order:
            key = col
            text = fmt_pct(row[key])
            cells.append(emphasize_latex(text, ranks[key].get(method, "")))
        total_text = f"{row['Total']*100:.1f} \\scriptsize{{$\\pm$ {row['Total_std']*100:.1f}}}"
        total_text = emphasize_latex(total_text, ranks["Total"].get(method, ""))
        label = METHOD_LABELS.get(method, method)
        if "ProtoTTA" in label:
            tex.append("\\rowcolor{gray!10}")
            label = f"\\textbf{{\\textit{{{label}}}}}"
        tex.append(f"{label} & " + " & ".join(cells) + f" & {total_text} \\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table*}\n")

    tex.append("\\begin{table*}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Efficiency and interpretability analysis on Stanford Dogs-C.}")
    tex.append("\\label{tab:dogs_efficiency}")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append("\\begin{tabular}{l|ccc|cc}")
    tex.append("\\toprule")
    tex.append("& \\multicolumn{3}{c|}{\\textbf{Interpretability Metrics}} & \\multicolumn{2}{c}{\\textbf{Efficiency Metrics}} \\\\")
    tex.append("\\textbf{Method} & \\textbf{PAC} $\\uparrow$ & \\textbf{PCA-W} $\\uparrow$ & \\textbf{Prediction Stability} $\\uparrow$ & \\textbf{Selection Rate} $\\downarrow$ & \\textbf{Rel. Speed} $\\uparrow$ \\\\")
    tex.append("\\midrule")
    tex.append("\\multicolumn{6}{c}{\\textit{\\textbf{Backbone: ProtoPFormer ~|~ Dataset: Stanford Dogs-C}}} \\\\")
    tex.append("\\midrule")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        if "ProtoTTA" in label:
            tex.append("\\rowcolor{gray!10}")
            label = f"\\textbf{{\\textit{{{label}}}}}"
        pac_txt = fmt_mean_std(pac[method]["mean"], pac[method]["std"]) if method in pac else "N/A"
        pca_txt = fmt_mean_std(pca_w[method]["mean"], pca_w[method]["std"]) if method in pca_w else "N/A"
        calib_txt = fmt_mean_std(calib[method]["mean"], calib[method]["std"]) if method in calib else "N/A"
        adapt_txt = f"{adapt[method]['mean']*100:.1f}\\%" if method in adapt else "N/A"
        speed_txt = f"{speed[method]['relative']:.1f}\\%" if method in speed and speed[method]["relative"] is not None else "N/A"
        tex.append(f"{label} & ${pac_txt}$ & ${pca_txt}$ & ${calib_txt}$ & {adapt_txt} & {speed_txt} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table*}\n")
    return "\n".join(tex)


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown and LaTeX reports from Dogs robustness JSON.")
    parser.add_argument("--input", default="robustness_results_dogs_full.json", help="Input JSON path")
    parser.add_argument("--severity", type=int, default=5, help="Severity key to report")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for .md and .tex files")
    args = parser.parse_args()

    input_path = Path(args.input)
    metadata, results = load_results(input_path)
    methods = available_methods(results, args.severity)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_suffix("")

    md = generate_markdown(metadata, results, methods, args.severity)
    tex = generate_latex(metadata, results, methods, args.severity)

    md_path = output_prefix.with_name(output_prefix.name + "_report.md")
    tex_path = output_prefix.with_name(output_prefix.name + "_tables.tex")
    md_path.write_text(md)
    tex_path.write_text(tex)

    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
