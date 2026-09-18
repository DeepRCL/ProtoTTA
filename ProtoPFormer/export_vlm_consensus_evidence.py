#!/usr/bin/env python3
"""Export frozen-versus-ProtoTTA evidence for the VLM consensus selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_robustness_dogs import load_corrupted_dataset, load_model, seed_everything
from noise_utils import CORRUPTION_TYPES
from proto_tta import setup_proto_tta


ROOT = Path(__file__).resolve().parent


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def humanize(folder: str) -> str:
    return folder.split("-", 1)[-1].replace("_", " ")


def local_activations(output) -> torch.Tensor:
    auxiliary = output[1]
    if len(auxiliary) >= 7:
        return auxiliary[5]
    if len(auxiliary) == 6:
        return auxiliary[4]
    raise ValueError(f"Unsupported ProtoPFormer output with {len(auxiliary)} auxiliaries")


def prototype_record(index: int, score: float, names: list[str], evidence_dir: Path) -> dict:
    class_index = index // 10
    return {
        "proto_idx": index,
        "class_index": class_index,
        "class_name": names[class_index],
        "contribution": score,
        "patch_locations": [[], []],
        "slots": [],
        "prototype_image_path": str(
            (evidence_dir / f"local_{index:04d}_overlay.jpg").resolve()
        ),
    }


def select_evidence(
    scores: torch.Tensor,
    prediction: int,
    names: list[str],
    evidence_dir: Path,
) -> tuple[list[dict], list[dict]]:
    class_indices = torch.arange(prediction * 10, prediction * 10 + 10)
    predicted_order = class_indices[torch.argsort(scores[class_indices], descending=True)[:5]]
    any_order = torch.argsort(scores, descending=True)[:10]
    predicted = [
        prototype_record(int(index), float(scores[index]), names, evidence_dir)
        for index in predicted_order
    ]
    any_class = [
        prototype_record(int(index), float(scores[index]), names, evidence_dir)
        for index in any_order
    ]
    return predicted, any_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corruption", choices=CORRUPTION_TYPES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "datasets/stanford_dogs_c"
    )
    parser.add_argument(
        "--prototype-evidence",
        type=Path,
        default=ROOT / "results/vlm_purified/evidence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/vlm_consensus_selector",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir / "export" / f"seed{args.seed}" / f"{args.corruption}.json"
    if output.exists() and not args.overwrite:
        print(f"Already complete: {output}")
        return
    seed_everything(args.seed)
    device = torch.device("cuda")
    adapted = setup_proto_tta(
        load_model(args.model, device),
        lr=1e-3,
        steps=1,
        use_importance=False,
        use_confidence=True,
        use_geometric_filter=True,
        geo_filter_threshold=0.55,
        consensus_strategy="max",
        consensus_ratio=0.5,
        adaptation_mode="layernorm_attn_bias",
        prototype_branch="both",
        similarity_mapping="sigmoid",
        sigmoid_center=1.0,
        sigmoid_temp=1.0,
        proto_weight=0.2,
        logit_weight=0.8,
        shared_confidence_weighting=True,
        gradient_normalize=True,
        record_diagnostics=True,
        record_vlm_evidence=True,
        model_mode="train",
    )
    loader = load_corrupted_dataset(
        args.data_dir, args.corruption, 5, args.batch_size, args.num_workers
    )
    dataset = loader.dataset
    names = [humanize(value) for value in dataset.classes]
    after_logits_batches, after_local_batches, label_batches = [], [], []
    for batch_index, (images, labels) in enumerate(
        tqdm(loader, desc=f"{args.corruption}:adapted")
    ):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        after_logits = adapted(images).detach().float().cpu()
        if not torch.allclose(after_logits, adapted.last_vlm_evidence["logits"]):
            raise RuntimeError("Returned logits do not match captured pre-update evidence")
        after_logits_batches.append(after_logits)
        after_local_batches.append(adapted.last_vlm_evidence["local_raw"])
        label_batches.append(labels.clone())
    after_logits_all = torch.cat(after_logits_batches)
    after_local_all = torch.cat(after_local_batches)
    labels_all = torch.cat(label_batches)

    # Run the frozen model only after the complete adaptation trajectory. This
    # makes the adapted pass byte-for-byte equivalent to the one-model paper
    # evaluator; evidence collection cannot perturb its RNG stream.
    del adapted
    torch.cuda.empty_cache()
    source = load_model(args.model, device).eval()
    source_loader = load_corrupted_dataset(
        args.data_dir, args.corruption, 5, args.batch_size, args.num_workers
    )
    records, evidence = {}, {}
    cursor = 0
    for batch_index, (images, labels) in enumerate(
        tqdm(source_loader, desc=f"{args.corruption}:frozen")
    ):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            source_output = source(images)
        before_logits = source_output[0].detach().float().cpu()
        before_local = local_activations(source_output).detach().float().cpu()
        stop = cursor + len(labels)
        after_logits = after_logits_all[cursor:stop]
        after_local = after_local_all[cursor:stop]
        if not torch.equal(labels, labels_all[cursor:stop]):
            raise RuntimeError("Frozen and adapted data passes are not aligned")
        before_prob = before_logits.softmax(dim=1)
        after_prob = after_logits.softmax(dim=1)
        before_pred = before_logits.argmax(dim=1)
        after_pred = after_logits.argmax(dim=1)
        for local_index, label in enumerate(labels.tolist()):
            sample_path, _ = dataset.samples[cursor + local_index]
            relative = str(Path(sample_path).relative_to(dataset.root))
            key = f"{args.corruption}::{relative}"
            bp, ap = int(before_pred[local_index]), int(after_pred[local_index])
            before_top2 = before_prob[local_index].topk(2).values
            after_top2 = after_prob[local_index].topk(2).values
            records[key] = {
                "sample_index": cursor + local_index,
                "image_path": relative,
                "ground_truth_index": int(label),
                "before_prediction_index": bp,
                "after_prediction_index": ap,
                "before_prediction": names[bp],
                "after_prediction": names[ap],
                "before_correct": bp == int(label),
                "after_correct": ap == int(label),
                "before_msp": float(before_top2[0]),
                "after_msp": float(after_top2[0]),
                "before_margin": float(before_top2[0] - before_top2[1]),
                "after_margin": float(after_top2[0] - after_top2[1]),
            }
            if bp != ap:
                before_predicted, before_any = select_evidence(
                    before_local[local_index], bp, names, args.prototype_evidence
                )
                after_predicted, after_any = select_evidence(
                    after_local[local_index], ap, names, args.prototype_evidence
                )
                evidence[key] = {
                    "before_predicted": before_predicted,
                    "before_any": before_any,
                    "after_predicted": after_predicted,
                    "after_any": after_any,
                }
        cursor += len(labels)
    expected = len(dataset) if args.max_batches is None else min(
        len(dataset), args.max_batches * args.batch_size
    )
    payload = {
        "schema_version": 1,
        "backbone": "ProtoPFormer",
        "configuration": "strongest fixed ProtoTTA lambda=0.2",
        "seed": args.seed,
        "corruption": args.corruption,
        "severity": 5,
        "complete": len(records) == expected,
        "num_dataset_samples": len(dataset),
        "num_records": len(records),
        "num_disagreements": len(evidence),
        "before_accuracy": float(np.mean([row["before_correct"] for row in records.values()])),
        "after_accuracy": float(np.mean([row["after_correct"] for row in records.values()])),
        "records": records,
        "evidence": evidence,
    }
    write_json(output, payload)
    print(json.dumps({key: payload[key] for key in (
        "corruption", "complete", "num_records", "num_disagreements",
        "before_accuracy", "after_accuracy"
    )}, indent=2))


if __name__ == "__main__":
    main()
