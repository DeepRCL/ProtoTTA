#!/usr/bin/env python3
"""Create paper-ready ProtoViT representation/controller visualizations.

The network is kept frozen.  Ground-truth labels are used only after feature
extraction to color points and quantify separability; none of the controller
signals uses labels or clean/source statistics.
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import model  # noqa: F401 -- required to unpickle the saved ProtoViT model
from preprocess import mean, std
from settings import img_size


DEFAULT_CORRUPTIONS = [
    "gaussian_noise", "fog", "gaussian_blur", "elastic_transform",
    "brightness", "jpeg_compression", "contrast", "defocus_blur",
    "frost", "impulse_noise", "pixelate", "shot_noise", "speckle_noise",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=(
        "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
    ))
    parser.add_argument("--clean-dir", default="datasets/cub200_cropped/test_cropped")
    parser.add_argument("--corrupt-root", default="datasets/cub200_c")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--controller-results", default=(
        "results/absolute_router_metrics/10523/cub200c_absolute_router_seed0.json"
    ))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--plot-classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_indices(targets, samples_per_class, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    selected = []
    for class_id in np.unique(targets):
        candidates = np.flatnonzero(targets == class_id)
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_class].tolist())
    return sorted(selected)


def make_loader(path, batch_size, workers, samples_per_class, seed):
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    indices = balanced_indices(dataset.targets, samples_per_class, seed)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    return dataset, loader


def consensus_topk_mean(similarities, ratio=0.5):
    if similarities.ndim == 2:
        return similarities
    k = max(1, int(similarities.shape[2] * ratio))
    return similarities.topk(k, dim=2).values.mean(dim=2)


def grouped_class_scores(similarities, prototype_classes, num_classes, topk=3):
    counts = torch.bincount(prototype_classes, minlength=num_classes)
    order = torch.argsort(prototype_classes)
    if counts.min() > 0 and torch.equal(counts, counts[0].expand_as(counts)):
        grouped = similarities[:, order].reshape(
            similarities.shape[0], num_classes, int(counts[0].item())
        )
        return grouped.topk(min(topk, grouped.shape[2]), dim=2).values.mean(dim=2)
    scores = []
    for class_id in range(num_classes):
        values = similarities[:, prototype_classes == class_id]
        if values.shape[1] == 0:
            scores.append(similarities.new_full((similarities.shape[0],), -1.0))
        else:
            scores.append(values.topk(min(topk, values.shape[1]), dim=1).values.mean(dim=1))
    return torch.stack(scores, dim=1)


@torch.inference_mode()
def extract(network, loader, device, description):
    prototype_classes = network.prototype_class_identity.argmax(dim=1).to(device)
    chunks = {key: [] for key in (
        "labels", "logits", "output_probs", "class_scores", "r_proto",
        "r_output", "lambda_relative", "lambda_activation", "max_similarity",
    )}
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        logits, _, subprototype_similarities = network(images)
        similarities = consensus_topk_mean(subprototype_similarities)
        positive = ((similarities.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(1e-8, 1.0)
        probs = logits.softmax(dim=1)
        class_scores = grouped_class_scores(
            positive, prototype_classes, logits.shape[1], topk=3
        )
        predicted = logits.argmax(dim=1)
        selected = class_scores.gather(1, predicted[:, None]).squeeze(1)
        competitor = class_scores.clone().scatter(
            1, predicted[:, None], float("-inf")
        ).max(dim=1).values
        r_proto = ((selected - competitor).clamp_min(0.0) /
                   selected.abs().clamp_min(1e-8)).clamp(0.0, 1.0)
        top2 = probs.topk(2, dim=1).values
        r_output = ((top2[:, 0] - top2[:, 1]) /
                    top2[:, 0].clamp_min(1e-8)).clamp(0.0, 1.0)
        lambda_relative = r_proto / (r_proto + r_output + 1e-8)

        target_mask = prototype_classes[None, :] == predicted[:, None]
        target_scores = positive.masked_fill(~target_mask, float("-inf"))
        top_target = target_scores.topk(3, dim=1).values
        lambda_activation = ((top_target - 0.5).abs().mean(dim=1) / 0.25).clamp(0.0, 1.0)

        values = {
            "labels": labels,
            "logits": logits,
            "output_probs": probs,
            "class_scores": class_scores,
            "r_proto": r_proto,
            "r_output": r_output,
            "lambda_relative": lambda_relative,
            "lambda_activation": lambda_activation,
            "max_similarity": similarities.max(dim=1).values,
        }
        for key, value in values.items():
            chunks[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def safe_silhouette(features, labels):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    try:
        return float(silhouette_score(features, labels, metric="cosine"))
    except ValueError:
        return float("nan")


def summarize(features):
    labels = features["labels"]
    output_pred = features["output_probs"].argmax(axis=1)
    prototype_pred = features["class_scores"].argmax(axis=1)
    return {
        "samples": int(labels.size),
        "output_accuracy": float(np.mean(output_pred == labels)),
        "prototype_evidence_accuracy": float(np.mean(prototype_pred == labels)),
        "output_silhouette": safe_silhouette(features["output_probs"], labels),
        "prototype_silhouette": safe_silhouette(features["class_scores"], labels),
        "mean_r_proto": float(features["r_proto"].mean()),
        "mean_r_output": float(features["r_output"].mean()),
        "mean_lambda_relative": float(features["lambda_relative"].mean()),
        "mean_lambda_activation": float(features["lambda_activation"].mean()),
        "reliable_fraction_at_0.92": float(np.mean(features["max_similarity"] > 0.92)),
    }


def select_farthest_classes(clean_features, count):
    labels = clean_features["labels"]
    evidence = StandardScaler().fit_transform(clean_features["class_scores"])
    classes = np.unique(labels)
    centroids = np.stack([evidence[labels == c].mean(axis=0) for c in classes])
    centroid_mean = centroids.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(centroids - centroid_mean, axis=1)))
    chosen = [first]
    while len(chosen) < min(count, len(classes)):
        distances = np.stack([
            np.linalg.norm(centroids - centroids[index], axis=1)
            for index in chosen
        ]).min(axis=0)
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return classes[np.asarray(chosen)]


def joint_embedding(clean_features, corrupt_features, key, selected_classes):
    clean_mask = np.isin(clean_features["labels"], selected_classes)
    corrupt_mask = np.isin(corrupt_features["labels"], selected_classes)
    clean_x = clean_features[key][clean_mask]
    corrupt_x = corrupt_features[key][corrupt_mask]
    all_x = StandardScaler().fit_transform(np.concatenate([clean_x, corrupt_x], axis=0))
    coords = PCA(n_components=2, random_state=0).fit_transform(all_x)
    split = clean_x.shape[0]
    return (
        coords[:split], clean_features["labels"][clean_mask],
        coords[split:], corrupt_features["labels"][corrupt_mask],
    )


def add_ellipse(ax, points, color):
    if len(points) < 3:
        return
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 3.0 * np.sqrt(eigenvalues)
    ax.add_patch(Ellipse(
        points.mean(axis=0), width, height, angle=angle,
        facecolor=color, edgecolor=color, alpha=0.10, linewidth=1.2,
    ))


def draw_embedding(ax, coords, labels, selected_classes, palette):
    for index, class_id in enumerate(selected_classes):
        points = coords[labels == class_id]
        color = palette[index]
        add_ellipse(ax, points, color)
        ax.scatter(points[:, 0], points[:, 1], s=22, color=color,
                   edgecolors="white", linewidths=0.45, alpha=0.90)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def save_figure(fig, stem, transparent=False):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=transparent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=400,
                transparent=transparent)
    plt.close(fig)


def make_embedding_assets(clean, corrupt, corruption, selected_classes, output_dir):
    palette = plt.get_cmap("tab10")(np.arange(len(selected_classes)))
    embedded = {}
    # Prototype class evidence and pre-softmax logits are used for geometry.
    # Softmax probabilities are intentionally not embedded: confident vectors
    # lie near simplex corners and yield unstable, ray-like PCA projections.
    for key in ("class_scores", "logits"):
        embedded[key] = joint_embedding(clean, corrupt, key, selected_classes)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.2))
    titles = (("Prototype", "Output"), ("Prototype", "Output"))
    for column, key in enumerate(("class_scores", "logits")):
        clean_xy, clean_y, corrupt_xy, corrupt_y = embedded[key]
        draw_embedding(axes[0, column], clean_xy, clean_y, selected_classes, palette)
        draw_embedding(axes[1, column], corrupt_xy, corrupt_y, selected_classes, palette)
        axes[0, column].set_title(titles[0][column], fontsize=10, weight="semibold")
    axes[0, 0].set_ylabel("Clean", fontsize=10, weight="semibold")
    axes[1, 0].set_ylabel(corruption.replace("_", " ").title(), fontsize=10,
                         weight="semibold")
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.7)
    save_figure(fig, output_dir / "embedding_comparison")

    # Text-free vector panels for compositing into the main method figure.
    for key, short_name in (("class_scores", "prototype"), ("logits", "output")):
        clean_xy, clean_y, corrupt_xy, corrupt_y = embedded[key]
        for domain, coords, labels in (
            ("clean", clean_xy, clean_y), ("corrupted", corrupt_xy, corrupt_y)
        ):
            fig, ax = plt.subplots(figsize=(2.5, 2.1))
            draw_embedding(ax, coords, labels, selected_classes, palette)
            fig.tight_layout(pad=0.05)
            save_figure(fig, output_dir / f"{short_name}_{domain}", transparent=True)


def load_controller_diagnostics(path):
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    results = payload.get("results", {})
    if not results:
        return {}
    method = next(iter(results.values()))
    diagnostics = {}
    for corruption, severities in method.items():
        entry = severities.get("5") or next(iter(severities.values()))
        stats = entry.get("adaptation_stats", {}) if entry else {}
        lambdas = stats.get("adaptive_lambda_raw", [])
        q_proto = stats.get("proto_gradient_consistency", [])
        q_output = stats.get("output_gradient_consistency", [])
        gates = stats.get("adaptive_router_gate", [])
        if lambdas:
            diagnostics[corruption] = {
                "lambda": float(np.mean(lambdas)),
                "q_proto": float(np.mean(q_proto)) if q_proto else float("nan"),
                "q_output": float(np.mean(q_output)) if q_output else float("nan"),
                "native_fraction": float(np.mean(gates)) if gates else float("nan"),
            }
    return diagnostics


def plot_controller_relation(summaries, controller_lambdas, output_dir):
    names, advantage, routed = [], [], []
    for name, values in summaries.items():
        if name == "clean" or name not in controller_lambdas:
            continue
        names.append(name)
        advantage.append(values["prototype_silhouette"] - values["output_silhouette"])
        routed.append(controller_lambdas[name])
    if not names:
        return
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    scatter = ax.scatter(advantage, routed, c=routed, cmap="viridis", s=35,
                         edgecolors="white", linewidths=0.5)
    for name, x, y in zip(names, advantage, routed):
        ax.annotate("".join(word[0] for word in name.split("_")), (x, y),
                    xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Prototype separation advantage", fontsize=8)
    ax.set_ylabel(r"Prototype weight $\lambda$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    save_figure(fig, output_dir / "controller_relation")


def plot_controller_signal_map(controller_diagnostics, output_dir):
    """Plot the exact label-free quantities used by the batch router."""
    if not controller_diagnostics:
        return
    names = list(controller_diagnostics)
    q_output = np.asarray([controller_diagnostics[n]["q_output"] for n in names])
    q_proto = np.asarray([controller_diagnostics[n]["q_proto"] for n in names])
    lambdas = np.asarray([controller_diagnostics[n]["lambda"] for n in names])
    fig, ax = plt.subplots(figsize=(3.25, 2.65))
    scatter = ax.scatter(q_output, q_proto, c=lambdas, cmap="viridis", s=42,
                         edgecolors="white", linewidths=0.55, vmin=0, vmax=1)
    x_limit = max(0.15, float(q_output.max()) * 1.15)
    x = np.linspace(0.0, x_limit, 100)
    ax.plot(x, 2.0 * x, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.axhline(0.25, color="#9ca3af", linewidth=0.9, linestyle=":")
    for name, x_value, y_value in zip(names, q_output, q_proto):
        short = "".join(word[0] for word in name.split("_"))
        ax.annotate(short, (x_value, y_value), xytext=(3, 2),
                    textcoords="offset points", fontsize=6)
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(0.0, max(0.65, float(q_proto.max()) * 1.10))
    ax.set_xlabel(r"Output consistency $q_O$", fontsize=8)
    ax.set_ylabel(r"Prototype consistency $q_P$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label(r"$\lambda$", fontsize=8)
    colorbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    save_figure(fig, output_dir / "controller_signal_map")


def plot_evidence_summary(summaries, representative, output_dir):
    labels = ["Clean", representative.replace("_", " ").title()]
    values = [summaries["clean"], summaries[representative]]
    proto = [v["mean_r_proto"] for v in values]
    output = [v["mean_r_output"] for v in values]
    x = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(3.0, 2.35))
    ax.bar(x - width / 2, proto, width, color="#63ad6f", label="Prototype")
    ax.bar(x + width / 2, output, width, color="#5689c7", label="Output")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Relative evidence", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout(pad=0.4)
    save_figure(fig, output_dir / "relative_evidence")


def plot_output_profile(features, stem, topk=20):
    """Export the mean ranked softmax profile without class-name clutter."""
    sorted_probs = np.sort(features["output_probs"], axis=1)[:, ::-1]
    topk = min(int(topk), sorted_probs.shape[1])
    heights = sorted_probs[:, :topk].mean(axis=0)
    colors = plt.get_cmap("Blues_r")(np.linspace(0.25, 0.78, topk))
    fig, ax = plt.subplots(figsize=(3.0, 1.65))
    ax.bar(np.arange(topk), heights, color=colors, width=0.78)
    ax.set_ylim(0.0, max(1.0, float(heights.max()) * 1.08))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    fig.tight_layout(pad=0.08)
    save_figure(fig, stem, transparent=True)


def plot_output_entropy_distribution(clean_features, corrupt_features, stem):
    """Plot the actual normalized output-entropy signal used by adaptation."""
    distributions = []
    for features in (clean_features, corrupt_features):
        probs = np.clip(features["output_probs"], 1e-12, 1.0)
        entropy = -(probs * np.log(probs)).sum(axis=1) / np.log(probs.shape[1])
        distributions.append(entropy)
    bins = np.linspace(0.0, max(0.65, max(x.max() for x in distributions)), 36)
    fig, ax = plt.subplots(figsize=(3.0, 1.85))
    ax.hist(distributions[0], bins=bins, density=True, histtype="step",
            linewidth=1.8, color="#315f9d", label="Clean")
    ax.hist(distributions[1], bins=bins, density=True, histtype="stepfilled",
            linewidth=1.2, color="#8eb0d9", alpha=0.45, label="Corrupted")
    ax.set_xlabel("Normalized output entropy", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    save_figure(fig, stem)


def make_output_logit_assets(clean, corrupt, selected_classes, output_dir):
    """Use pre-softmax logits for output-space geometry (not sparse probabilities)."""
    palette = plt.get_cmap("tab10")(np.arange(len(selected_classes)))
    clean_xy, clean_y, corrupt_xy, corrupt_y = joint_embedding(
        clean, corrupt, "logits", selected_classes
    )
    for domain, coords, labels in (
        ("clean", clean_xy, clean_y), ("corrupted", corrupt_xy, corrupt_y)
    ):
        fig, ax = plt.subplots(figsize=(2.5, 2.1))
        draw_embedding(ax, coords, labels, selected_classes, palette)
        fig.tight_layout(pad=0.05)
        save_figure(fig, output_dir / f"output_logits_{domain}", transparent=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    network = torch.load(args.model, map_location=device, weights_only=False).to(device)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    clean_dataset, clean_loader = make_loader(
        args.clean_dir, args.batch_size, args.workers,
        args.samples_per_class, args.seed,
    )
    print(f"clean samples={len(clean_loader.dataset)} classes={len(clean_dataset.classes)}")
    clean = extract(network, clean_loader, device, "clean")
    all_features = {"clean": clean}
    summaries = {"clean": summarize(clean)}

    corrupt_root = Path(args.corrupt_root)
    for index, corruption in enumerate(args.corruptions):
        path = corrupt_root / corruption / str(args.severity)
        if not path.exists():
            print(f"warning: missing {path}; skipping")
            continue
        _, loader = make_loader(
            path, args.batch_size, args.workers,
            args.samples_per_class, args.seed,
        )
        features = extract(network, loader, device, corruption)
        all_features[corruption] = features
        summaries[corruption] = summarize(features)

    # Select the corruption with the largest drop in prototype-space
    # separability. This criterion is fixed, recorded, and label-based only for
    # analysis/visualization (never for adaptation).
    available = [name for name in args.corruptions if name in summaries]
    representative = min(
        available, key=lambda name: summaries[name]["prototype_silhouette"]
    )
    selected_classes = select_farthest_classes(clean, args.plot_classes)
    controller_diagnostics = load_controller_diagnostics(args.controller_results)
    controller_lambdas = {
        name: values["lambda"] for name, values in controller_diagnostics.items()
    }
    for name, value in controller_lambdas.items():
        if name in summaries:
            summaries[name]["mean_routed_lambda_from_adaptive_run"] = value

    make_embedding_assets(
        clean, all_features[representative], representative,
        selected_classes, output_dir,
    )
    plot_controller_relation(summaries, controller_lambdas, output_dir)
    plot_controller_signal_map(controller_diagnostics, output_dir)
    plot_evidence_summary(summaries, representative, output_dir)
    plot_output_profile(clean, output_dir / "output_profile_clean", topk=20)
    plot_output_profile(
        all_features[representative], output_dir / "output_profile_corrupted", topk=20
    )
    plot_output_entropy_distribution(
        clean, all_features[representative], output_dir / "output_entropy_distribution"
    )
    make_output_logit_assets(
        clean, all_features[representative], selected_classes, output_dir
    )

    np.savez_compressed(
        output_dir / "representation_features.npz",
        **{
            f"{domain}__{key}": value
            for domain, features in all_features.items()
            for key, value in features.items()
        },
    )
    audit = {
        "checkpoint": str(Path(args.model).resolve()),
        "clean_dir": str(Path(args.clean_dir).resolve()),
        "corrupt_root": str(Path(args.corrupt_root).resolve()),
        "severity": args.severity,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "selected_class_ids": selected_classes.astype(int).tolist(),
        "selected_class_names": [clean_dataset.classes[int(i)] for i in selected_classes],
        "class_selection": "farthest clean prototype-evidence centroids",
        "representative_corruption": representative,
        "representative_selection": "lowest prototype-space silhouette among evaluated corruptions",
        "labels_policy": (
            "Labels are used only for plotting colors, class selection, and post-hoc "
            "separability metrics; extraction and controller evidence are label-free."
        ),
        "summaries": summaries,
        "controller_results": str(Path(args.controller_results).resolve()),
        "controller_diagnostics": controller_diagnostics,
    }
    (output_dir / "figure_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "representative_corruption": representative,
        "selected_classes": audit["selected_class_names"],
    }, indent=2))


if __name__ == "__main__":
    main()
