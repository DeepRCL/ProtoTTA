#!/usr/bin/env python3
"""Evaluate offline-VLM semantic guidance on ProtoTTA over CUB-200-C."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import vlm_eval
from evaluate_robustness import load_corrupted_dataset, seed_everything, setup_proto_entropy
from semantic_prototype_guidance import load_quality_vector, load_student, write_json


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("semantic_prototta_eval")
CORRUPTIONS = vlm_eval.CORRUPTION_TYPES
VARIANTS = ["baseline_control", "static", "distilled", "combined", "combined_consistency", "combined_oracle"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS + ["purified_main"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/cub200_c")
    parser.add_argument("--guidance-dir", type=Path, default=ROOT / "results/semantic_prototta")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--rollback-tolerance", type=float, default=0.0)
    parser.add_argument("--semantic-logit-blend", type=float, default=0.5)
    parser.add_argument("--semantic-fusion", choices=["fixed", "confidence_guard"], default="fixed")
    parser.add_argument("--semantic-contrast-weight", type=float, default=0.5)
    parser.add_argument("--semantic-temperature", type=float, default=0.25)
    return parser.parse_args()


def logits_of(outputs):
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def snapshot(wrapper):
    return {
        "parameters": {name: value.detach().clone() for name, value in wrapper.model.named_parameters() if value.requires_grad},
        "optimizer": copy.deepcopy(wrapper.optimizer.state_dict()),
    }


def restore(wrapper, state):
    with torch.no_grad():
        named = dict(wrapper.model.named_parameters())
        for name, value in state["parameters"].items():
            named[name].copy_(value)
    wrapper.optimizer.load_state_dict(state["optimizer"])


def consistency_score(logits_a, logits_b):
    probs_a, probs_b = logits_a.float().softmax(1), logits_b.float().softmax(1)
    classes = logits_a.shape[1]
    confidence = 0.5 * (probs_a.max(1).values + probs_b.max(1).values)
    entropy = 0.5 * (
        -(probs_a * probs_a.clamp_min(1e-8).log()).sum(1)
        -(probs_b * probs_b.clamp_min(1e-8).log()).sum(1)
    ) / np.log(classes)
    agreement = logits_a.argmax(1).eq(logits_b.argmax(1)).float()
    return (0.5 * agreement + 0.3 * (1.0 - entropy) + 0.2 * confidence).mean()


def setup(args, device, corruption):
    vlm_eval.install_timm_checkpoint_compat()
    vlm_eval.lazy_import_runtime_modules()
    base = torch.load(args.model, weights_only=False).to(device).eval()
    use_static = args.variant in ("static", "combined", "combined_consistency", "combined_oracle", "purified_main")
    use_student = args.variant in ("distilled", "combined", "combined_consistency", "combined_oracle")
    quality_name = "prototype_quality_ranked.json" if args.variant == "purified_main" else "prototype_quality.json"
    quality = load_quality_vector(args.guidance_dir / quality_name) if use_static else None
    # Distilled-only still uses a neutral vector for feature computation.
    feature_quality = quality if quality is not None else torch.ones(base.prototype_class_identity.shape[0])
    student_path = args.guidance_dir / "compatibility_students_loco" / f"{corruption}.json"
    student = load_student(student_path) if use_student else None
    return setup_proto_entropy(
        base,
        use_importance=True,
        use_confidence=True,
        reset_mode=None,
        use_geometric_filter=True,
        geo_filter_threshold=0.92,
        consensus_strategy="top_k_mean",
        consensus_ratio=0.5,
        adaptation_mode="layernorm_attn_bias",
        use_ensemble_entropy=False,
        semantic_prototype_weights=quality if use_static else feature_quality,
        compatibility_student=student,
        compatibility_floor=0.25,
        # Match the canonical ProtoTTA configuration used by the paper
        # baseline. With a pure prototype objective this controls the actual
        # update magnitude and therefore cannot be omitted from an ablation.
        shared_confidence_weighting=True,
        gradient_normalize=True,
        record_diagnostics=True,
        semantic_logit_blend=args.semantic_logit_blend if args.variant == "purified_main" else 0.0,
        semantic_fusion=args.semantic_fusion,
        semantic_contrast_weight=args.semantic_contrast_weight if args.variant == "purified_main" else 0.0,
        semantic_temperature=args.semantic_temperature,
    )


def evaluate_corruption(args, corruption, device):
    seed_everything(args.seed)
    wrapper = setup(args, device, corruption)
    loader = load_corrupted_dataset(args.data_dir, corruption, 5, args.batch_size)
    correct = total = post_correct = 0
    commits = rollbacks = oracle_harmful = 0
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc=f"{args.variant}:{corruption}")):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        state = snapshot(wrapper) if args.variant.endswith(("consistency", "oracle")) else None
        if args.variant == "combined_consistency":
            with torch.no_grad():
                pre_a = logits_of(wrapper.forward_no_adapt(images))
                pre_b = logits_of(wrapper.forward_no_adapt(torch.flip(images, dims=[3])))
                pre_score = consistency_score(pre_a, pre_b)
            outputs = wrapper(images)
            with torch.no_grad():
                post_a = logits_of(wrapper.forward_no_adapt(images))
                post_b = logits_of(wrapper.forward_no_adapt(torch.flip(images, dims=[3])))
                post_score = consistency_score(post_a, post_b)
            if post_score + args.rollback_tolerance < pre_score:
                restore(wrapper, state)
                rollbacks += 1
            else:
                commits += 1
            logits = logits_of(outputs)
            post_logits = post_a
        elif args.variant == "combined_oracle":
            with torch.no_grad():
                pre_logits = logits_of(wrapper.forward_no_adapt(images))
            outputs = wrapper(images)
            with torch.no_grad():
                post_logits = logits_of(wrapper.forward_no_adapt(images))
            pre_hits = pre_logits.argmax(1).eq(labels).sum()
            post_hits = post_logits.argmax(1).eq(labels).sum()
            if post_hits < pre_hits:
                restore(wrapper, state)
                rollbacks += 1
                oracle_harmful += 1
            else:
                commits += 1
            logits = logits_of(outputs)
        else:
            outputs = wrapper(images)
            logits = logits_of(outputs)
            with torch.no_grad():
                post_logits = logits_of(wrapper.forward_no_adapt(images))
        correct += int(logits.argmax(1).eq(labels).sum().item())
        post_correct += int(post_logits.argmax(1).eq(labels).sum().item())
        total += int(labels.numel())
    return {
        "online_accuracy": 100.0 * correct / total,
        "immediate_post_update_accuracy": 100.0 * post_correct / total,
        "correct": correct,
        "total": total,
        "commit_batches": commits,
        "rollback_batches": rollbacks,
        "rollback_rate": rollbacks / max(commits + rollbacks, 1),
        "oracle_harmful_batches": oracle_harmful,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.output is None:
        args.output = args.guidance_dir / "evaluation" / f"{args.variant}_seed{args.seed}.json"
    corruptions = CORRUPTIONS if "all" in args.corruptions else args.corruptions
    unknown = set(corruptions) - set(CORRUPTIONS)
    if unknown:
        raise ValueError(f"Unknown corruptions: {sorted(unknown)}")
    payload = {
        "schema_version": 1,
        "variant": args.variant,
        "seed": args.seed,
        "severity": 5,
        "stream_protocol": "fresh checkpoint per corruption; continuous within-corruption batch adaptation",
        "label_usage": "none except combined_oracle, whose labels only select commit/rollback",
        "model": str(args.model),
        "semantic_logit_blend": args.semantic_logit_blend if args.variant == "purified_main" else 0.0,
        "semantic_fusion": args.semantic_fusion,
        "semantic_contrast_weight": args.semantic_contrast_weight if args.variant == "purified_main" else 0.0,
        "semantic_temperature": args.semantic_temperature,
        "results": {},
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    for corruption in corruptions:
        if corruption in payload["results"]:
            continue
        started = time.time()
        result = evaluate_corruption(args, corruption, device)
        result["elapsed_seconds"] = time.time() - started
        payload["results"][corruption] = result
        write_json(args.output, payload)
        LOGGER.info("%s %s online=%.3f post=%.3f", args.variant, corruption,
                    result["online_accuracy"], result["immediate_post_update_accuracy"])


if __name__ == "__main__":
    main()
