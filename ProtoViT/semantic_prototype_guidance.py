"""Shared features and artifacts for semantic ProtoTTA guidance.

The VLM is used offline.  At test time this module only evaluates fixed
prototype-quality weights and a tiny linear student, so no labels or VLM calls
are part of the adaptation loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


FEATURE_NAMES = [
    "msp",
    "softmax_margin",
    "target_activation_max",
    "target_activation_mean",
    "target_activation_std",
    "target_contribution_max",
    "target_contribution_mean",
    "competitor_activation_max",
    "activation_margin",
    "top10_target_fraction",
    "top10_class_concentration",
    "top10_class_entropy",
    "target_quality_activation_mean",
    "top10_quality_activation_mean",
    "target_quality_min",
]


def read_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_quality_vector(path: Path, expected: int = 2000) -> torch.Tensor:
    payload = read_json(path)
    values = payload.get("prototype_quality", payload)
    if isinstance(values, dict):
        values = [values[str(index)] for index in range(expected)]
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() != expected:
        raise ValueError(f"Expected {expected} prototype scores, got {tuple(tensor.shape)}")
    return tensor.clamp(0.05, 1.0)


def load_student(path: Path) -> Dict:
    payload = read_json(path)
    if payload["feature_names"] != FEATURE_NAMES:
        raise ValueError("Compatibility-student feature schema does not match runtime")
    for key in ("mean", "scale", "weights"):
        if len(payload[key]) != len(FEATURE_NAMES):
            raise ValueError(f"Bad student {key} length")
    return payload


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.maximum(weights, 0.0)
    denominator = float(weights.sum())
    return float((values * weights).sum() / denominator) if denominator > 1e-12 else float(values.mean())


def cached_features(entry: Dict, msp: float, margin: float, quality: Sequence[float]) -> np.ndarray:
    """Feature vector from the saved reasoning-board metadata."""
    target = entry["predicted_top_prototypes"]
    any_top = entry["any_class_top_prototypes"][:10]
    predicted = int(entry["predicted_index"])
    target_a = np.asarray([float(item["activation"]) for item in target], dtype=np.float64)
    target_c = np.asarray([abs(float(item["contribution"])) for item in target], dtype=np.float64)
    any_a = np.asarray([float(item["activation"]) for item in any_top], dtype=np.float64)
    any_c = np.asarray([abs(float(item["contribution"])) for item in any_top], dtype=np.float64)
    any_classes = np.asarray([int(item["class_index"]) for item in any_top], dtype=np.int64)
    target_q = np.asarray([quality[int(item["proto_idx"])] for item in target], dtype=np.float64)
    any_q = np.asarray([quality[int(item["proto_idx"])] for item in any_top], dtype=np.float64)
    competitor = any_a[any_classes != predicted]
    counts = np.bincount(any_classes, minlength=200).astype(np.float64)
    probs = counts[counts > 0] / max(float(len(any_top)), 1.0)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(max(len(any_top), 2)))
    return np.asarray([
        msp,
        margin,
        target_a.max(),
        target_a.mean(),
        target_a.std(),
        target_c.max(),
        target_c.mean(),
        competitor.max() if competitor.size else target_a.max(),
        target_a.max() - (competitor.max() if competitor.size else target_a.max()),
        float((any_classes == predicted).mean()),
        float(counts.max() / max(float(len(any_top)), 1.0)),
        entropy,
        _weighted_average(target_q, target_a),
        _weighted_average(any_q, any_a),
        target_q.min(),
    ], dtype=np.float32)


def runtime_features(
    logits: torch.Tensor,
    similarities: torch.Tensor,
    prototype_classes: torch.Tensor,
    last_layer_weight: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Differentiation-free counterpart of :func:`cached_features`."""
    with torch.no_grad():
        probs = logits.float().softmax(dim=1)
        top2 = probs.topk(2, dim=1).values
        predicted = logits.argmax(dim=1)
        batch, num_prototypes = similarities.shape
        target_mask = prototype_classes.unsqueeze(0).eq(predicted.unsqueeze(1))
        target_a_all = similarities.masked_fill(~target_mask, float("-inf"))
        k_target = min(5, int(target_mask[0].sum().item()))
        target_a, target_idx = target_a_all.topk(k_target, dim=1)
        class_weights = last_layer_weight[predicted].abs()
        contributions = similarities * class_weights
        target_c = contributions.gather(1, target_idx).abs()
        k_any = min(10, num_prototypes)
        any_c, any_idx = contributions.topk(k_any, dim=1)
        any_a = similarities.gather(1, any_idx)
        any_classes = prototype_classes[any_idx]
        any_target = any_classes.eq(predicted.unsqueeze(1))
        competitor = any_a.masked_fill(any_target, float("-inf")).max(dim=1).values
        competitor = torch.where(torch.isfinite(competitor), competitor, target_a[:, 0])
        one_hot = torch.nn.functional.one_hot(any_classes, num_classes=logits.shape[1]).float()
        class_counts = one_hot.sum(dim=1)
        class_probs = class_counts / float(k_any)
        class_entropy = -(class_probs * (class_probs + 1e-12).log()).sum(dim=1) / np.log(max(k_any, 2))
        target_q = quality[target_idx]
        any_q = quality[any_idx]
        target_q_mean = (target_q * target_a.clamp_min(0)).sum(dim=1) / target_a.clamp_min(0).sum(dim=1).clamp_min(1e-8)
        any_q_mean = (any_q * any_a.clamp_min(0)).sum(dim=1) / any_a.clamp_min(0).sum(dim=1).clamp_min(1e-8)
        return torch.stack([
            top2[:, 0],
            top2[:, 0] - top2[:, 1],
            target_a[:, 0],
            target_a.mean(dim=1),
            target_a.std(dim=1, unbiased=False),
            target_c.max(dim=1).values,
            target_c.mean(dim=1),
            competitor,
            target_a[:, 0] - competitor,
            any_target.float().mean(dim=1),
            class_counts.max(dim=1).values / float(k_any),
            class_entropy,
            target_q_mean,
            any_q_mean,
            target_q.min(dim=1).values,
        ], dim=1)


def predict_student(features: torch.Tensor, student: Dict) -> torch.Tensor:
    mean = features.new_tensor(student["mean"])
    scale = features.new_tensor(student["scale"])
    weights = features.new_tensor(student["weights"])
    prediction = (features - mean) / scale * weights
    return (prediction.sum(dim=1) + float(student["bias"])).clamp(0.0, 1.0)
