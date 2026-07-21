#!/usr/bin/env python3
"""
Visualize the ProtoTTA lambda sweep results for all models.

Each model was swept over lambda in {0.0, 0.3, 0.5, 0.7, 1.0}, where lambda
blends the adaptation objective between:
    lambda = 0.0 -> pure logit entropy (standard TTA, e.g. Tent-style)
    lambda = 1.0 -> pure prototype entropy (ProtoTTA)
    0 < lambda < 1 -> ProtoTTA+ blend of both signals

Results were produced by the per-model `slurm_lambda_sweep.sh` scripts and are
stored under `<Model>/results/lambda_sweep/*.json`.

Usage:
    python visualize_lambda_sweep.py [--output-dir lambda_sweep_analysis]

Outputs (under --output-dir, default "lambda_sweep_analysis/"):
    per_model_accuracy_vs_lambda.png   -- one subplot per model: mean accuracy
                                           (+ per-corruption lines) vs lambda
    combined_relative_comparison.png   -- all models on one axis, accuracy
                                           relative to each model's lambda=0.0
    corruption_heatmaps.png            -- corruption x lambda heatmap per model
    best_lambda_summary.png            -- bar chart of best lambda per model
    overall_summary.csv                -- mean accuracy per (model, lambda)
    per_corruption.csv                 -- mean accuracy per (model, lambda, corruption)
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9

REPO_ROOT = Path(__file__).resolve().parent

# Each entry describes where a model's lambda-sweep JSON files live and how
# to pull results out of them ("results" -> mode_key -> corruption -> severity).
MODEL_CONFIGS = [
    {
        "key": "protovit",
        "name": "ProtoViT",
        "dataset": "CUB-200-C",
        "dir": REPO_ROOT / "ProtoViT" / "results" / "lambda_sweep",
        "file_glob": "cub200c_lambda*.json",
        "mode_key": "proto_imp_conf_v3",
    },
    {
        "key": "protopformer",
        "name": "ProtoPFormer",
        "dataset": "Stanford Dogs-C",
        "dir": REPO_ROOT / "ProtoPFormer" / "results" / "lambda_sweep",
        "file_glob": "dogs_c_lambda*.json",
        "mode_key": "proto_tta",
    },
    {
        "key": "protosvit",
        "name": "ProtoSViT",
        "dataset": "Stanford Cars-C",
        "dir": REPO_ROOT / "protosvit" / "results" / "lambda_sweep",
        "file_glob": "cars_c_lambda*.json",
        "mode_key": "proto_tta",
    },
    {
        "key": "protolens",
        "name": "ProtoLens",
        "dataset": "Amazon-C",
        "dir": REPO_ROOT / "ProtoLens" / "results" / "lambda_sweep",
        "file_glob": "amazon_c_lambda*.json",
        "mode_key": "prototta",
    },
]

LAMBDA_RE = re.compile(r"lambda([0-9]+\.?[0-9]*)\.json$")


def load_model_records(cfg: dict) -> pd.DataFrame:
    """Load all lambda-sweep JSON files for a single model into long-form rows."""
    records = []
    if not cfg["dir"].is_dir():
        print(f"  [skip] {cfg['name']}: no directory at {cfg['dir']}")
        return pd.DataFrame(records)

    files = sorted(cfg["dir"].glob(cfg["file_glob"]))
    if not files:
        print(f"  [skip] {cfg['name']}: no files matching {cfg['file_glob']} in {cfg['dir']}")
        return pd.DataFrame(records)

    for f in files:
        m = LAMBDA_RE.search(f.name)
        if not m:
            continue
        lam = float(m.group(1))

        with open(f) as fh:
            data = json.load(fh)

        mode_results = data.get("results", {}).get(cfg["mode_key"], {})
        for corruption, sev_dict in mode_results.items():
            for severity, metrics in sev_dict.items():
                acc = metrics.get("accuracy") if isinstance(metrics, dict) else None
                if acc is None:
                    continue
                acc_pct = acc * 100 if acc <= 1.0 else acc
                records.append(
                    {
                        "model": cfg["name"],
                        "dataset": cfg["dataset"],
                        "lambda": lam,
                        "corruption": corruption,
                        "severity": str(severity),
                        "accuracy": acc_pct,
                    }
                )

    df = pd.DataFrame(records)
    n_lambdas = df["lambda"].nunique() if not df.empty else 0
    print(f"  [ok]   {cfg['name']}: {len(files)} files, {n_lambdas} lambda values, {len(df)} rows")
    return df


def load_all() -> pd.DataFrame:
    print("Loading lambda sweep results:")
    frames = [load_model_records(cfg) for cfg in MODEL_CONFIGS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No lambda sweep results found for any model.")
    return pd.concat(frames, ignore_index=True)


def plot_per_model_accuracy_vs_lambda(df: pd.DataFrame, out_path: Path):
    """One subplot per model: faint per-corruption lines + bold mean line vs lambda."""
    models = [cfg["name"] for cfg in MODEL_CONFIGS if cfg["name"] in df["model"].unique()]
    n = len(models)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["model"] == model]
        dataset = sub["dataset"].iloc[0]

        # Faint line per corruption (averaged over severity if multiple)
        per_corr = sub.groupby(["corruption", "lambda"])["accuracy"].mean().reset_index()
        for corruption, cgroup in per_corr.groupby("corruption"):
            cgroup = cgroup.sort_values("lambda")
            ax.plot(cgroup["lambda"], cgroup["accuracy"], color="gray", alpha=0.35, linewidth=1)

        overall = sub.groupby("lambda")["accuracy"].mean().reset_index().sort_values("lambda")
        ax.plot(
            overall["lambda"], overall["accuracy"],
            color="crimson", linewidth=2.5, marker="o", markersize=7,
            label="Mean over corruptions",
        )

        best_row = overall.loc[overall["accuracy"].idxmax()]
        ax.scatter([best_row["lambda"]], [best_row["accuracy"]], color="darkgreen", s=110, zorder=5,
                   label=f"Best λ={best_row['lambda']:.1f} ({best_row['accuracy']:.1f}%)")

        ax.axvline(0.0, color="steelblue", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(1.0, color="darkorange", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(f"{model} ({dataset})")
        ax.set_xlabel("λ  (0 = logit entropy, 1 = prototype entropy)")
        ax.set_ylabel("Mean accuracy (%)")
        ax.legend(loc="best")

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Accuracy vs λ per model (gray lines = individual corruptions)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_combined_relative(df: pd.DataFrame, out_path: Path):
    """All models overlaid, normalized to each model's own lambda=0.0 accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=len(MODEL_CONFIGS))

    for color, cfg in zip(palette, MODEL_CONFIGS):
        model = cfg["name"]
        if model not in df["model"].unique():
            continue
        sub = df[df["model"] == model]
        overall = sub.groupby("lambda")["accuracy"].mean().reset_index().sort_values("lambda")
        if 0.0 not in overall["lambda"].values:
            continue
        baseline = overall.loc[overall["lambda"] == 0.0, "accuracy"].iloc[0]
        overall["relative"] = overall["accuracy"] - baseline
        ax.plot(
            overall["lambda"], overall["relative"],
            marker="o", markersize=7, linewidth=2.2, color=color,
            label=f"{model} ({cfg['dataset']})",
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("λ  (0 = logit entropy, 1 = prototype entropy)")
    ax.set_ylabel("Accuracy change vs λ=0.0 (percentage points)")
    ax.set_title("Effect of λ relative to pure logit-entropy baseline (λ=0.0)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_corruption_heatmaps(df: pd.DataFrame, out_path: Path):
    """Corruption x lambda accuracy heatmap, one panel per model."""
    models = [cfg["name"] for cfg in MODEL_CONFIGS if cfg["name"] in df["model"].unique()]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        pivot = sub.groupby(["corruption", "lambda"])["accuracy"].mean().unstack("lambda")
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        sns.heatmap(
            pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
            cbar_kws={"label": "Accuracy (%)"}, linewidths=0.5,
        )
        ax.set_title(model)
        ax.set_xlabel("λ")
        ax.set_ylabel("Corruption")

    fig.suptitle("Per-corruption accuracy (%) across λ", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_best_lambda_summary(df: pd.DataFrame, out_path: Path):
    """Bar chart: logit-entropy (λ=0) vs prototype-entropy (λ=1) vs best λ, per model."""
    models = [cfg["name"] for cfg in MODEL_CONFIGS if cfg["name"] in df["model"].unique()]
    rows = []
    for model in models:
        overall = df[df["model"] == model].groupby("lambda")["accuracy"].mean()
        best_lam = overall.idxmax()
        rows.append({"model": model, "λ=0.0 (logit entropy)": overall.get(0.0, np.nan)})
        rows[-1]["λ=1.0 (prototype entropy)"] = overall.get(1.0, np.nan)
        rows[-1][f"best λ={best_lam:.1f}"] = overall.max()

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(models))
    width = 0.25

    logit_vals = [r["λ=0.0 (logit entropy)"] for r in rows]
    proto_vals = [r["λ=1.0 (prototype entropy)"] for r in rows]
    best_vals = [r[k] for r, k in zip(rows, [k for row in rows for k in row if k.startswith("best")])]
    best_labels = [k for row in rows for k in row if k.startswith("best")]

    ax.bar(x - width, logit_vals, width, label="λ=0.0 (logit entropy)", color="steelblue")
    ax.bar(x, proto_vals, width, label="λ=1.0 (prototype entropy)", color="darkorange")
    ax.bar(x + width, best_vals, width, label="Best λ", color="darkgreen")

    for i, lbl in enumerate(best_labels):
        ax.text(i + width, best_vals[i] + 0.5, lbl.replace("best ", ""), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_title("Logit entropy vs prototype entropy vs best λ, per model")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=str, default=str(REPO_ROOT / "lambda_sweep_analysis"),
        help="Directory to write plots and CSV summaries to.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all()

    overall_summary = (
        df.groupby(["model", "dataset", "lambda"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values(["model", "lambda"])
    )
    overall_summary.to_csv(out_dir / "overall_summary.csv", index=False)

    per_corruption = (
        df.groupby(["model", "dataset", "lambda", "corruption"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values(["model", "corruption", "lambda"])
    )
    per_corruption.to_csv(out_dir / "per_corruption.csv", index=False)

    print(f"\nSaved summary CSVs to {out_dir}")
    print("\nOverall mean accuracy (%) by model / lambda:")
    print(overall_summary.pivot(index="model", columns="lambda", values="accuracy").round(2).to_string())

    print("\nGenerating plots:")
    plot_per_model_accuracy_vs_lambda(df, out_dir / "per_model_accuracy_vs_lambda.png")
    plot_combined_relative(df, out_dir / "combined_relative_comparison.png")
    plot_corruption_heatmaps(df, out_dir / "corruption_heatmaps.png")
    plot_best_lambda_summary(df, out_dir / "best_lambda_summary.png")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
