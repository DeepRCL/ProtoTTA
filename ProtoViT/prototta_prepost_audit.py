#!/usr/bin/env python3
"""Audit the causal meaning and oracle headroom of individual ProtoTTA updates.

For batch t, PRE is the current continuously-adapted state before applying the
batch-t update and POST is the same state after that tentative update.  The
paper's standard prediction is PRE: ProtoEntropy returns logits computed before
its optimizer step.  Labels are used only for post-hoc measurements.

The audit also evaluates batch t+1 from both the PRE and POST snapshots.  This
measures whether committing update t helps or harms the next batch without
changing the fixed ProtoTTA trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import fullset_vlm_gate as gate


LOGGER = logging.getLogger("prototta_prepost_audit")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "prototta_prepost_audit")
    parser.add_argument("--data-dir", type=Path, default=root / "datasets" / "cub200_c")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "saved_models" / "deit_small_patch16_224" / "exp1" / "14finetuned0.8609.pth",
    )
    parser.add_argument("--corruption", choices=gate.CORRUPTIONS, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def stats(output, labels) -> Dict:
    import torch

    value = logits(output)
    probability = torch.softmax(value.float(), dim=1)
    top2 = probability.topk(2, dim=1).values
    prediction = value.argmax(dim=1)
    return {
        "correct": int((prediction == labels).sum().item()),
        "n": int(labels.numel()),
        "mean_msp": float(top2[:, 0].mean().item()),
        "mean_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
        "prediction": prediction.detach().cpu(),
    }


def move(batch: Tuple, device):
    images, labels = batch
    return images.to(device), labels.to(device)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.model = args.model.resolve()

    import torch
    import proto_entropy
    import vlm_eval
    import evaluate_robustness as robustness

    vlm_eval.install_timm_checkpoint_compat()
    vlm_eval.lazy_import_runtime_modules()
    robustness.seed_everything(args.seed)
    device = torch.device("cuda")
    dataset = vlm_eval.load_corruption_dataset(args.data_dir, args.corruption, 5)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    base = torch.load(str(args.model), weights_only=False).to(device)
    adapted = gate.exact_fixed_prototta(base)
    adapted.eval()

    iterator = iter(loader)
    current = next(iterator, None)
    batch_index = 0
    records = []
    while current is not None and (args.max_batches is None or batch_index < args.max_batches):
        following = next(iterator, None)
        images, labels = move(current, device)
        with torch.no_grad():
            pre_output = adapted.forward_no_adapt(images)
        pre = stats(pre_output, labels)
        pre_model, pre_optimizer = proto_entropy.copy_model_and_optimizer(adapted.model, adapted.optimizer)

        # ProtoEntropy returns PRE logits and then commits one optimizer step.
        returned_output = adapted(images)
        returned = stats(returned_output, labels)
        with torch.no_grad():
            post_output = adapted.forward_no_adapt(images)
        post = stats(post_output, labels)
        post_model, post_optimizer = proto_entropy.copy_model_and_optimizer(adapted.model, adapted.optimizer)

        record = {
            "batch_index": batch_index,
            "n": pre["n"],
            "pre_correct": pre["correct"],
            "returned_correct": returned["correct"],
            "post_correct": post["correct"],
            "returned_matches_pre_predictions": bool(torch.equal(pre["prediction"], returned["prediction"])),
            "pre_post_prediction_changes": int((pre["prediction"] != post["prediction"]).sum().item()),
            "pre_mean_msp": pre["mean_msp"],
            "post_mean_msp": post["mean_msp"],
            "pre_mean_margin": pre["mean_margin"],
            "post_mean_margin": post["mean_margin"],
        }
        if following is not None:
            next_images, next_labels = move(following, device)
            with torch.no_grad():
                commit_output = adapted.forward_no_adapt(next_images)
            commit = stats(commit_output, next_labels)
            proto_entropy.load_model_and_optimizer(
                adapted.model, adapted.optimizer, pre_model, pre_optimizer
            )
            with torch.no_grad():
                rollback_output = adapted.forward_no_adapt(next_images)
            rollback = stats(rollback_output, next_labels)
            proto_entropy.load_model_and_optimizer(
                adapted.model, adapted.optimizer, post_model, post_optimizer
            )
            record.update(
                {
                    "next_n": commit["n"],
                    "next_commit_correct": commit["correct"],
                    "next_rollback_correct": rollback["correct"],
                    "next_prediction_changes": int(
                        (commit["prediction"] != rollback["prediction"]).sum().item()
                    ),
                    "update_outcome_next_batch": (
                        "beneficial" if commit["correct"] > rollback["correct"]
                        else "harmful" if commit["correct"] < rollback["correct"]
                        else "neutral"
                    ),
                }
            )
        records.append(record)
        LOGGER.info(
            "%s batch=%d pre=%d post=%d changes=%d next_delta=%s",
            args.corruption,
            batch_index,
            pre["correct"],
            post["correct"],
            record["pre_post_prediction_changes"],
            (
                record.get("next_commit_correct", 0) - record.get("next_rollback_correct", 0)
                if following is not None else "NA"
            ),
        )
        del pre_model, pre_optimizer, post_model, post_optimizer
        current = following
        batch_index += 1

    total_n = sum(item["n"] for item in records)
    transitions = [item for item in records if "next_commit_correct" in item]
    outcome_counts = {
        name: sum(item["update_outcome_next_batch"] == name for item in transitions)
        for name in ("beneficial", "harmful", "neutral")
    }
    payload = {
        "schema_version": 1,
        "corruption": args.corruption,
        "severity": 5,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "complete": args.max_batches is None and total_n == len(dataset),
        "num_samples": total_n,
        "num_batches": len(records),
        "paper_prediction_semantics": "PRE logits; optimizer step is applied after those logits are computed",
        "pre_accuracy": sum(item["pre_correct"] for item in records) / total_n,
        "post_same_batch_accuracy": sum(item["post_correct"] for item in records) / total_n,
        "returned_predictions_always_match_pre": all(
            item["returned_matches_pre_predictions"] for item in records
        ),
        "total_pre_post_prediction_changes": sum(
            item["pre_post_prediction_changes"] for item in records
        ),
        "next_batch_update_outcomes": outcome_counts,
        "next_batch_commit_correct": sum(item["next_commit_correct"] for item in transitions),
        "next_batch_rollback_correct": sum(item["next_rollback_correct"] for item in transitions),
        "next_batch_oracle_correct": sum(
            max(item["next_commit_correct"], item["next_rollback_correct"])
            for item in transitions
        ),
        "records": records,
    }
    destination = args.output_dir / f"{args.corruption}.json"
    write_json(destination, payload)
    LOGGER.info("Wrote %s", destination)


if __name__ == "__main__":
    main()
