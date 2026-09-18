#!/usr/bin/env python3
"""Evaluate source-audited VLM purification on the strongest ProtoPFormer TTA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from evaluate_robustness_dogs import load_corrupted_dataset, load_model, seed_everything
from noise_utils import CORRUPTION_TYPES
from proto_tta import setup_proto_tta

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/stanford_dogs_c")
    parser.add_argument("--quality", type=Path, default=ROOT / "results/vlm_purified/prototype_quality_ranked.json")
    parser.add_argument("--neutral-quality", action="store_true",
                        help="Use all-one weights for an implementation smoke test only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--semantic-logit-blend", type=float, default=0.5)
    parser.add_argument("--semantic-fusion", choices=["fixed", "confidence_guard"], default="fixed")
    parser.add_argument("--semantic-output-only", action="store_true",
                        help="Preserve native ProtoTTA updates; use VLM only for guarded prediction fusion")
    parser.add_argument("--semantic-quality-normalization",
                        choices=["none", "class_mean"], default="none")
    parser.add_argument("--semantic-contrast-weight", type=float, default=0.5)
    parser.add_argument("--semantic-temperature", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    return parser.parse_args()


def main():
    args = parse_args()
    quality = ({"local_quality": [1.0] * 1200, "global_quality": [1.0] * 600}
               if args.neutral_quality else
               json.loads(args.quality.read_text(encoding="utf-8")))
    corruptions = CORRUPTION_TYPES if "all" in args.corruptions else args.corruptions
    payload = {
        "schema_version": 1,
        "method": "VLM-Purified ProtoPFormer ProtoTTA",
        "seed": args.seed,
        "severity": 5,
        "stream_protocol": "fresh checkpoint per corruption; continuous within-corruption adaptation",
        "label_usage": "source-training prototype identity only; no target labels",
        "base_proto_lambda": 0.2,
        "semantic_logit_blend": args.semantic_logit_blend,
        "semantic_fusion": args.semantic_fusion,
        "semantic_output_only": args.semantic_output_only,
        "semantic_quality_normalization": args.semantic_quality_normalization,
        "semantic_contrast_weight": args.semantic_contrast_weight,
        "semantic_temperature": args.semantic_temperature,
        "results": {},
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("seed", "base_proto_lambda", "semantic_logit_blend", "semantic_fusion",
                    "semantic_output_only",
                    "semantic_quality_normalization",
                    "semantic_contrast_weight", "semantic_temperature"):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"Refusing incompatible resume: {key}")
        payload = existing
    device = torch.device("cuda")
    for corruption in corruptions:
        if corruption in payload["results"]:
            continue
        seed_everything(args.seed)
        model = load_model(args.model, device)
        adapted = setup_proto_tta(
            model, lr=1e-3, steps=1, use_importance=False, use_confidence=True,
            use_geometric_filter=True, geo_filter_threshold=0.55,
            consensus_strategy="max", consensus_ratio=0.5,
            adaptation_mode="layernorm_attn_bias", prototype_branch="both",
            similarity_mapping="sigmoid", sigmoid_center=1.0, sigmoid_temp=1.0,
            proto_weight=0.2, logit_weight=0.8,
            shared_confidence_weighting=True, gradient_normalize=True,
            record_diagnostics=True, model_mode="train",
            semantic_local_weights=quality["local_quality"],
            semantic_global_weights=quality["global_quality"],
            semantic_logit_blend=args.semantic_logit_blend,
            semantic_fusion=args.semantic_fusion,
            semantic_output_only=args.semantic_output_only,
            semantic_quality_normalization=args.semantic_quality_normalization,
            semantic_contrast_weight=(0.0 if args.semantic_output_only else args.semantic_contrast_weight),
            semantic_temperature=args.semantic_temperature,
        )
        loader = load_corrupted_dataset(args.data_dir, corruption, 5,
                                        args.batch_size, args.num_workers)
        correct = total = 0
        started = time.time()
        for batch_index, (images, labels) in enumerate(tqdm(loader, desc=corruption)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = adapted(images)
            correct += int(logits.argmax(1).eq(labels).sum().item())
            total += int(labels.numel())
        payload["results"][corruption] = {
            "online_accuracy": 100.0 * correct / total,
            "correct": correct, "total": total,
            "elapsed_seconds": time.time() - started,
        }
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
