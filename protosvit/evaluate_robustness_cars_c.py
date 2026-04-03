#!/usr/bin/env python3
"""
Comprehensive robustness evaluation for ProtoS-ViT on Stanford Cars-C.

Features:
- Additive / resumable JSON output
- Per-corruption, per-severity results for all requested methods
- Table-ready aggregate metrics:
  - PAC (semantic consistency)
  - PCA-W (prototype alignment, weighted)
  - Prediction stability
  - Selection rate
  - Relative speed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import run_inference_cars_c as ric


DEFAULT_METHODS = [
    "unadapted",
    "tent",
    "eata",
    "sar",
    "prototta",
    "prototta_plus_70_30",
    "prototta_plus_80_20",
    "prototta_plus_90_10",
]


METHOD_CONFIGS = {
    "unadapted": {
        "display_name": "Unadapted",
        "kind": "normal",
    },
    "tent": {
        "display_name": "Tent",
        "kind": "tent",
    },
    "eata": {
        "display_name": "EATA",
        "kind": "eata",
    },
    "sar": {
        "display_name": "SAR",
        "kind": "sar",
    },
    "prototta": {
        "display_name": "ProtoTTA",
        "kind": "proto_tta",
        "proto_weight": 1.0,
        "logit_weight": 0.0,
    },
    "prototta_plus_70_30": {
        "display_name": "ProtoTTA+ (70/30)",
        "kind": "proto_tta_plus",
        "proto_weight": 0.7,
        "logit_weight": 0.3,
    },
    "prototta_plus_80_20": {
        "display_name": "ProtoTTA+ (80/20)",
        "kind": "proto_tta_plus",
        "proto_weight": 0.8,
        "logit_weight": 0.2,
    },
    "prototta_plus_90_10": {
        "display_name": "ProtoTTA+ (90/10)",
        "kind": "proto_tta_plus",
        "proto_weight": 0.9,
        "logit_weight": 0.1,
    },
}


def to_python(value):
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    if isinstance(value, tuple):
        return [to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


class EfficiencyTracker:
    def __init__(self, method_name: str, device: torch.device):
        self.method_name = method_name
        self.device = device
        self.total_time = 0.0
        self.num_samples = 0
        self.batch_times: List[float] = []
        self.num_adapted_params = 0
        self.total_params = 0
        self.num_adaptation_steps = 0

    def count_adapted_parameters(self, model: torch.nn.Module, adapted_params: Optional[Iterable[torch.nn.Parameter]] = None):
        self.total_params = sum(p.numel() for p in model.parameters())
        if adapted_params is None:
            self.num_adapted_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            self.num_adapted_params = sum(p.numel() for p in adapted_params if p.requires_grad)

    @contextmanager
    def track_inference(self, batch_size: int):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            self.total_time += elapsed
            self.num_samples += batch_size
            self.batch_times.append(elapsed)

    def record_adaptation_step(self, num_steps: int):
        self.num_adaptation_steps += num_steps

    def get_metrics(self) -> Dict[str, float]:
        return {
            "method_name": self.method_name,
            "total_time_sec": self.total_time,
            "num_samples": self.num_samples,
            "time_per_sample_ms": (self.total_time / max(self.num_samples, 1)) * 1000.0,
            "avg_batch_time_ms": (np.mean(self.batch_times) * 1000.0) if self.batch_times else 0.0,
            "std_batch_time_ms": (np.std(self.batch_times) * 1000.0) if self.batch_times else 0.0,
            "throughput_samples_per_sec": self.num_samples / max(self.total_time, 1e-8),
            "num_adapted_params": self.num_adapted_params,
            "total_params": self.total_params,
            "adaptation_ratio": self.num_adapted_params / max(self.total_params, 1),
            "num_adaptation_steps": self.num_adaptation_steps,
        }


class CarsPrototypeMetricsEvaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device, top_k: int = 10):
        self.device = device
        self.top_k = top_k
        self.model = model
        self.class_weights = model.classification_head.weight.detach().cpu()
        self.proto_identities = self.class_weights.argmax(dim=0)
        self.clean_activations: Optional[torch.Tensor] = None
        self.clean_logits: Optional[torch.Tensor] = None
        self.clean_predictions: Optional[torch.Tensor] = None
        self.clean_labels: Optional[torch.Tensor] = None

    def collect_clean_baseline(self, loader):
        activations = []
        logits = []
        predictions = []
        labels_all = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                out = self.model(images)
                batch_logits = out["pred"].detach().cpu()
                batch_activations = out["similarity_score"].detach().cpu()
                activations.append(batch_activations)
                logits.append(batch_logits)
                predictions.append(batch_logits.argmax(dim=1))
                labels_all.append(labels.cpu())

        self.clean_activations = torch.cat(activations, dim=0)
        self.clean_logits = torch.cat(logits, dim=0)
        self.clean_predictions = torch.cat(predictions, dim=0)
        self.clean_labels = torch.cat(labels_all, dim=0)

    def _compute_pac(self, adapted_activations: torch.Tensor) -> Dict[str, float]:
        if self.clean_activations is None:
            raise ValueError("Clean baseline not collected.")
        n = min(len(adapted_activations), len(self.clean_activations))
        sims = F.cosine_similarity(
            adapted_activations[:n].float(),
            self.clean_activations[:n].float(),
            dim=1,
        ).cpu().numpy()
        return {
            "PAC_mean": float(np.mean(sims)),
            "PAC_std": float(np.std(sims)),
        }

    def _compute_pca_weighted(self, adapted_activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        labels = labels.cpu()
        scores = []
        for idx in range(len(labels)):
            activ = adapted_activations[idx].cpu()
            label = labels[idx].item()
            k = min(self.top_k, activ.numel())
            top_vals, top_idx = torch.topk(activ, k=k)
            importance = self.class_weights[label, top_idx].abs()
            contribution = top_vals * importance
            denom = contribution.sum().item()
            if denom <= 0:
                scores.append(0.0)
                continue
            correct_mask = (self.proto_identities[top_idx] == label).float()
            score = (contribution * correct_mask).sum().item() / denom
            scores.append(score)
        scores_np = np.asarray(scores)
        return {
            "PCA_weighted_mean": float(np.mean(scores_np)),
            "PCA_weighted_std": float(np.std(scores_np)),
        }

    def _compute_prediction_stability(self, adapted_predictions: torch.Tensor, adapted_logits: torch.Tensor) -> Dict[str, float]:
        if self.clean_predictions is None or self.clean_logits is None:
            raise ValueError("Clean baseline not collected.")
        n = min(len(adapted_predictions), len(self.clean_predictions))
        agreement = (adapted_predictions[:n].cpu() == self.clean_predictions[:n]).float().mean().item()
        logit_corr = F.cosine_similarity(
            adapted_logits[:n].float().cpu(),
            self.clean_logits[:n].float().cpu(),
            dim=1,
        ).mean().item()
        return {
            "prediction_stability": float(agreement),
            "prediction_stability_logit_corr": float(logit_corr),
        }

    def summarize(self, adapted_activations: torch.Tensor, adapted_logits: torch.Tensor, adapted_predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        metrics.update(self._compute_pac(adapted_activations))
        metrics.update(self._compute_pca_weighted(adapted_activations, labels))
        metrics.update(self._compute_prediction_stability(adapted_predictions, adapted_logits))
        return metrics


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json_atomic(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(to_python(payload), f, indent=2)
    os.replace(tmp, path)


def get_clean_loader(clean_dir: str, batch_size: int, num_workers: int):
    transform = transforms.Compose([
        transforms.Resize((ric.IMG_SIZE, ric.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(ric.MEAN, ric.STD),
    ])
    root = Path(clean_dir)
    candidates = [root / "val", root / "test", root]
    chosen = None
    for candidate in candidates:
        if candidate.is_dir():
            chosen = candidate
            break
    if chosen is None:
        raise FileNotFoundError(f"No clean Stanford Cars directory found under {clean_dir}")
    dataset = datasets.ImageFolder(str(chosen), transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def discover_corruptions_and_severities(
    cars_c_dir: str,
    severities: Optional[List[int]] = None,
    corruptions: Optional[List[str]] = None,
) -> List[Tuple[str, int]]:
    root = Path(cars_c_dir)
    combos = []
    allowed_severities = set(severities) if severities is not None else None
    allowed_corruptions = set(corruptions) if corruptions is not None else None
    for corruption in ric.CORRUPTION_TYPES:
        if allowed_corruptions is not None and corruption not in allowed_corruptions:
            continue
        corr_dir = root / corruption
        if not corr_dir.exists():
            continue
        severities = sorted(
            int(p.name) for p in corr_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
        for severity in severities:
            if allowed_severities is not None and severity not in allowed_severities:
                continue
            combos.append((corruption, severity))
    return combos


def build_method(method_name: str, ckpt: str, device: torch.device, loader, args, fishers_cache: Dict[str, Dict]):
    model = ric.load_model(ckpt, device)
    config = METHOD_CONFIGS[method_name]
    kind = config["kind"]

    if kind == "normal":
        return model
    if kind == "tent":
        return ric.setup_tent(model, lr=args.lr, steps=args.steps, adapt_mode=args.adapt_mode)
    if kind == "sar":
        return ric.setup_sar(model, lr=args.lr, steps=args.steps, adapt_mode=args.adapt_mode)
    if kind == "eata":
        fisher_key = f"{args.adapt_mode}:{args.use_clean_fisher}:{args.clean_fisher_samples}"
        fishers = fishers_cache.get(fisher_key)
        if fishers is None:
            fisher_loader = loader
            if args.use_clean_fisher:
                clean_loader = get_clean_loader(args.clean_dir, batch_size=min(args.batch_size, 32), num_workers=args.num_workers)
                fisher_loader = clean_loader
            ric.configure_model(model, args.adapt_mode)
            fishers = ric.compute_fishers(
                model,
                fisher_loader,
                device,
                adapt_mode=args.adapt_mode,
                num_samples=args.clean_fisher_samples,
            )
            fishers_cache[fisher_key] = fishers
        return ric.setup_eata(model, lr=args.lr, steps=args.steps, adapt_mode=args.adapt_mode, fishers=fishers)
    if kind in {"proto_tta", "proto_tta_plus"}:
        return ric.setup_proto_tta(
            model,
            lr=args.proto_lr,
            steps=args.steps,
            use_importance=True,
            use_confidence=True,
            geo_filter_threshold=args.proto_threshold,
            proto_weight=config.get("proto_weight", 1.0),
            logit_weight=config.get("logit_weight", 0.0),
            adapt_mode=args.adapt_mode,
        )
    raise ValueError(f"Unsupported method: {method_name}")


def evaluate_method(method_name: str, eval_model, loader, metrics_evaluator: CarsPrototypeMetricsEvaluator, device: torch.device) -> Dict:
    if hasattr(eval_model, "model"):
        base_model = eval_model.model
        adapted_params = [p for p in base_model.parameters() if p.requires_grad]
    else:
        base_model = eval_model
        adapted_params = []

    tracker = EfficiencyTracker(method_name, device)
    tracker.count_adapted_parameters(base_model, adapted_params=adapted_params)

    all_logits = []
    all_activations = []
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with tracker.track_inference(labels.size(0)):
            out = eval_model(images)
        logits = out["pred"]
        activations = out["similarity_score"]
        predictions = logits.argmax(dim=1)

        all_logits.append(logits.detach().cpu())
        all_activations.append(activations.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_predictions.append(predictions.detach().cpu())

    if hasattr(eval_model, "adaptation_stats"):
        stats = dict(eval_model.adaptation_stats)
    else:
        total = sum(x.size(0) for x in all_labels)
        stats = {"total_samples": total, "adapted_samples": 0, "total_updates": 0}

    logits = torch.cat(all_logits, dim=0)
    activations = torch.cat(all_activations, dim=0)
    labels = torch.cat(all_labels, dim=0)
    predictions = torch.cat(all_predictions, dim=0)

    if metrics_evaluator.clean_labels is not None and len(labels) == len(metrics_evaluator.clean_labels):
        if not torch.equal(labels.cpu(), metrics_evaluator.clean_labels.cpu()):
            raise RuntimeError(
                "Clean and corrupted label orders do not match. "
                "PAC / prediction stability would be invalid."
            )

    adapt_steps = getattr(eval_model, "steps", 0 if method_name == "unadapted" else 1)
    tracker.record_adaptation_step(len(loader) * adapt_steps)

    accuracy = predictions.eq(labels).float().mean().item()
    metrics = metrics_evaluator.summarize(activations, logits, predictions, labels)
    metrics["accuracy"] = float(accuracy)
    metrics["adaptation_stats"] = stats
    metrics["selection_rate"] = float(stats["adapted_samples"] / max(stats["total_samples"], 1))
    metrics["avg_updates_per_sample"] = float(stats["total_updates"] / max(stats["total_samples"], 1))
    metrics["efficiency"] = tracker.get_metrics()
    return metrics


def aggregate_results(results: Dict, methods: List[str]) -> Dict[str, Dict]:
    aggregates: Dict[str, Dict] = {}
    method_results = results.get("results", {})

    for method in methods:
        per_combo = []
        for corruption_map in method_results.get(method, {}).values():
            for severity_result in corruption_map.values():
                if severity_result is not None:
                    per_combo.append(severity_result)

        if not per_combo:
            aggregates[method] = {}
            continue

        metrics = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in per_combo])),
            "accuracy_std": float(np.std([r["accuracy"] for r in per_combo])),
            "PAC_mean": float(np.mean([r["PAC_mean"] for r in per_combo])),
            "PAC_std": float(np.std([r["PAC_mean"] for r in per_combo])),
            "PCA_weighted_mean": float(np.mean([r["PCA_weighted_mean"] for r in per_combo])),
            "PCA_weighted_std": float(np.std([r["PCA_weighted_mean"] for r in per_combo])),
            "prediction_stability_mean": float(np.mean([r["prediction_stability"] for r in per_combo])),
            "prediction_stability_std": float(np.std([r["prediction_stability"] for r in per_combo])),
            "selection_rate_mean": float(np.mean([r["selection_rate"] for r in per_combo])),
            "avg_updates_per_sample_mean": float(np.mean([r["avg_updates_per_sample"] for r in per_combo])),
            "time_per_sample_ms_mean": float(np.mean([r["efficiency"]["time_per_sample_ms"] for r in per_combo])),
        }

        rel_speeds = []
        method_map = method_results.get(method, {})
        normal_map = method_results.get("unadapted", {})
        for corruption, sev_map in method_map.items():
            for severity, result in sev_map.items():
                if result is None:
                    continue
                base_result = normal_map.get(corruption, {}).get(severity)
                if base_result is None:
                    continue
                base_t = base_result["efficiency"]["time_per_sample_ms"]
                curr_t = result["efficiency"]["time_per_sample_ms"]
                rel_speeds.append(base_t / max(curr_t, 1e-8))
        metrics["relative_speed_mean"] = float(np.mean(rel_speeds)) if rel_speeds else 1.0

        aggregates[method] = metrics

    return aggregates


def print_summary(aggregates: Dict[str, Dict], methods: List[str]):
    print("\n" + "=" * 110)
    print("TABLE-READY SUMMARY")
    print("=" * 110)
    header = (
        f"{'Method':<24}"
        f"{'PAC':>14}"
        f"{'PCA-W':>14}"
        f"{'Stability':>14}"
        f"{'Select %':>14}"
        f"{'RelSpeed %':>14}"
    )
    print(header)
    print("-" * len(header))
    for method in methods:
        row = aggregates.get(method, {})
        if not row:
            print(f"{method:<24}{'pending':>14}")
            continue
        print(
            f"{METHOD_CONFIGS[method]['display_name']:<24}"
            f"{row['PAC_mean']*100:>9.2f} ± {row['PAC_std']*100:<4.2f}"
            f"{row['PCA_weighted_mean']*100:>9.2f} ± {row['PCA_weighted_std']*100:<4.2f}"
            f"{row['prediction_stability_mean']*100:>9.2f} ± {row['prediction_stability_std']*100:<4.2f}"
            f"{row['selection_rate_mean']*100:>13.2f}"
            f"{row['relative_speed_mean']*100:>13.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for ProtoS-ViT on Cars-C")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--cars_c_dir", required=True, help="Cars-C root directory")
    parser.add_argument("--clean_dir", required=True, help="Clean Stanford Cars folder root (expects val/ or test/)")
    parser.add_argument("--output", default="results/cars_c_robustness_full.json", help="Additive JSON output path")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHOD_CONFIGS.keys()))
    parser.add_argument("--override_methods", nargs="*", default=[], choices=list(METHOD_CONFIGS.keys()))
    parser.add_argument("--severity", nargs="+", type=int, default=[5], help="Only evaluate these severity levels (default: 5)")
    parser.add_argument("--corruptions", nargs="+", default=None, choices=ric.CORRUPTION_TYPES, help="Optional subset of corruption types")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proto_lr", type=float, default=1e-4)
    parser.add_argument("--proto_threshold", type=float, default=0.9)
    parser.add_argument("--adapt_mode", default="vit", choices=[
        "vit",
        "layernorm",
        "layernorm_conv",
        "layernorm_proto",
        "layernorm_conv_proto",
        "layernorm_conv_proto_project",
        "vit_layernorm_conv_proto_project",
        "full_head",
    ])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--clean_fisher_samples", type=int, default=500)
    parser.add_argument("--use_clean_fisher", action="store_true", default=False)
    args = parser.parse_args()

    if args.adapt_mode == "full_head":
        args.adapt_mode = "vit_layernorm_conv_proto_project_head"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    existing = load_json(output_path)

    results = existing if existing else {
        "metadata": {
            "ckpt": args.ckpt,
            "cars_c_dir": args.cars_c_dir,
            "clean_dir": args.clean_dir,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": {},
    }
    results["metadata"].update({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "adapt_mode": args.adapt_mode,
        "steps": args.steps,
        "proto_threshold": args.proto_threshold,
        "methods": args.methods,
        "use_clean_fisher": args.use_clean_fisher,
        "severity": args.severity,
        "corruptions": args.corruptions,
    })

    clean_loader = get_clean_loader(args.clean_dir, args.batch_size, args.num_workers)
    clean_model = ric.load_model(args.ckpt, device)
    evaluator = CarsPrototypeMetricsEvaluator(clean_model, device=device)
    evaluator.collect_clean_baseline(clean_loader)

    combos = discover_corruptions_and_severities(
        args.cars_c_dir,
        severities=args.severity,
        corruptions=args.corruptions,
    )
    if not combos:
        raise FileNotFoundError(
            f"No Cars-C corruptions found under {args.cars_c_dir} "
            f"for severities={args.severity} and corruptions={args.corruptions}"
        )

    fishers_cache: Dict[str, Dict] = {}

    print("=" * 90)
    print("PROTOS-VIT ROBUSTNESS EVALUATION")
    print("=" * 90)
    print(f"Checkpoint     : {args.ckpt}")
    print(f"Cars-C dir     : {args.cars_c_dir}")
    print(f"Clean dir      : {args.clean_dir}")
    print(f"Methods        : {args.methods}")
    print(f"Combinations   : {len(combos)}")
    print(f"Severities     : {args.severity}")
    print(f"Corruptions    : {args.corruptions if args.corruptions is not None else 'all'}")
    print(f"Adapt mode     : {args.adapt_mode}")
    print(f"Device         : {device}")
    print("=" * 90)

    for method in args.methods:
        results["results"].setdefault(method, {})

    total_jobs = len(args.methods) * len(combos)
    done = 0

    for method in args.methods:
        for corruption, severity in combos:
            severity_key = str(severity)
            method_bucket = results["results"].setdefault(method, {})
            corr_bucket = method_bucket.setdefault(corruption, {})
            needs_override = method in set(args.override_methods)
            if severity_key in corr_bucket and corr_bucket[severity_key] is not None and not needs_override:
                done += 1
                continue

            loader = ric.get_loader(corruption, severity, args.cars_c_dir, args.batch_size, args.num_workers)
            clean_classes = getattr(clean_loader.dataset, "classes", None)
            corr_classes = getattr(loader.dataset, "classes", None)
            if clean_classes is not None and corr_classes is not None and clean_classes != corr_classes:
                raise RuntimeError(f"Class order mismatch between clean set and {corruption}/sev{severity}")

            print(f"\n[{done + 1}/{total_jobs}] {method} | {corruption} | severity {severity}")
            eval_model = build_method(method, args.ckpt, device, loader, args, fishers_cache)
            result = evaluate_method(method, eval_model, loader, evaluator, device)
            result["method"] = method
            result["corruption"] = corruption
            result["severity"] = severity
            corr_bucket[severity_key] = result

            results["aggregates"] = aggregate_results(results, args.methods)
            save_json_atomic(output_path, results)
            done += 1

            print(
                f"  acc={result['accuracy']*100:.2f}% | "
                f"PAC={result['PAC_mean']*100:.2f} | "
                f"PCA-W={result['PCA_weighted_mean']*100:.2f} | "
                f"stab={result['prediction_stability']*100:.2f}% | "
                f"select={result['selection_rate']*100:.2f}% | "
                f"time={result['efficiency']['time_per_sample_ms']:.2f} ms/sample"
            )

    results["aggregates"] = aggregate_results(results, args.methods)
    save_json_atomic(output_path, results)
    print_summary(results["aggregates"], args.methods)


if __name__ == "__main__":
    main()
