#!/usr/bin/env python3
"""Export sealed ProtoLens BEFORE/AFTER streams for label-free LLM scoring."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

# Set deterministic backend behavior before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from protolens_llm_common import (
    CLASS_NAMES, atomic_write_json, reject_label_fields, sha256_lines,
    stream_dir, task_for_index,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "llm_prototta"
DEFAULT_MODEL = ROOT / "log_folder" / "Yelp" / "_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005" / "model.pth"


def representative_phrases(model: Any, index: int) -> list[str]:
    values = np.asarray(model.sentence_pool[index]).reshape(-1).tolist()
    output = []
    for value in values:
        phrase = " ".join(str(value).split())
        if phrase and phrase.lower() != "none" and phrase not in output:
            output.append(phrase)
        if len(output) == 3:
            break
    return output


def prototype_evidence(model: Any, similarities: Any, prediction: int) -> list[dict]:
    import torch

    weights = model.fc.weight[prediction].detach()
    contribution = similarities.detach() * weights
    count = min(5, contribution.numel())
    indices = torch.topk(contribution.abs(), k=count).indices.tolist()
    output = []
    for index in indices:
        value = float(contribution[index].item())
        output.append({
            "prototype_index": int(index),
            "phrases": representative_phrases(model, index),
            "similarity": float(similarities[index].item()),
            "class_weight": float(weights[index].item()),
            "contribution": value,
            "effect": "supports prediction" if value >= 0 else "opposes prediction",
        })
    return output


def unpack(result: Any):
    output = result[0] if isinstance(result, tuple) else result
    if not isinstance(result, tuple) or len(result) < 4:
        raise RuntimeError("ProtoLens did not return prototype similarities")
    similarities = result[3]
    if similarities.ndim == 3:
        similarities = similarities.mean(dim=1)
    if similarities.ndim != 2:
        raise RuntimeError(f"unexpected similarity shape {tuple(similarities.shape)}")
    return output, similarities


def run(args: argparse.Namespace) -> Path:
    import torch
    import evaluate_robustness_amazonc as evaluator

    corruption, severity = task_for_index(args.task_id)
    destination = stream_dir(args.output_dir, corruption, severity)
    marker = destination / "export.complete.json"
    if marker.is_file() and not args.overwrite:
        print(f"reuse={marker}")
        return marker

    evaluator.seed_everything(args.random_state)
    evaluator.cfg.OPTIM.LR = 5e-6
    evaluator.cfg.OPTIM.STEPS = 1
    evaluator.cfg.MODEL.EPISODIC = False
    device = torch.device("cuda")
    frame = evaluator.load_corrupted_data(str(args.data_dir), corruption, severity)
    source, tokenizer, model_args = evaluator.load_model(str(args.model_path), device)
    adapted_base, _, _ = evaluator.load_model(str(args.model_path), device)
    source.eval()
    adapted = evaluator.setup_prototta(
        adapted_base, "layernorm_attn_bias", True, 0.1,
        importance_mode="global", sigmoid_temperature=5.0,
        logit_weight=0.5, adaptive_lambda=True,
        samplewise_lambda=True, gradient_normalize=True,
        adaptive_delta0=0.25, adaptive_topk=3,
        router_min_consistency=0.25, lambda_ema_momentum=0.0,
        lambda_min=0.0, lambda_max=1.0, record_diagnostics=True,
        adaptive_lambda_strategy="source_free_router_coverage_absolute",
    )
    adapted_model = adapted.model
    loader = evaluator.create_dataloader(
        frame, tokenizer, model_args.max_length, args.batch_size
    )
    records = []
    targets = []
    cursor = 0
    for batch_index, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        special_tokens_mask = batch["special_tokens_mask"].to(device)
        with torch.no_grad():
            before_result = source(
                input_ids=input_ids, attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask, mode="test",
                original_text=batch["original_text"],
                current_batch_num=batch_index,
            )
        after_result = adapted(
            input_ids=input_ids, attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask, mode="test",
            original_text=batch["original_text"],
            current_batch_num=batch_index,
        )
        before_logits, before_sim = unpack(before_result)
        after_logits, after_sim = unpack(after_result)
        before_prob = torch.softmax(before_logits.float(), dim=1)
        after_prob = torch.softmax(after_logits.float(), dim=1)
        before_pred = before_prob.argmax(dim=1)
        after_pred = after_prob.argmax(dim=1)
        labels = batch["label"].tolist()
        for local, text in enumerate(batch["original_text"]):
            index = cursor + local
            before = int(before_pred[local].item())
            after = int(after_pred[local].item())
            before_top = torch.topk(before_prob[local], k=2).values
            after_top = torch.topk(after_prob[local], k=2).values
            changed = before != after
            record = {
                "sample_id": f"sample-{index:04d}",
                "stream_index": index,
                "changed": changed,
                "before_prediction_index": before,
                "after_prediction_index": after,
                "before_prediction": CLASS_NAMES[before],
                "after_prediction": CLASS_NAMES[after],
                "before_msp": float(before_top[0].item()),
                "after_msp": float(after_top[0].item()),
                "before_margin": float((before_top[0] - before_top[1]).item()),
                "after_margin": float((after_top[0] - after_top[1]).item()),
            }
            if changed:
                record.update({
                    "corrupted_review": str(text),
                    "before_evidence": prototype_evidence(source, before_sim[local], before),
                    "after_evidence": prototype_evidence(adapted_model, after_sim[local], after),
                })
            records.append(record)
            targets.append(int(labels[local]))
        cursor += len(labels)
        print(
            f"{corruption}-{severity} batch={batch_index} "
            f"records={len(records)} disagreements={sum(r['changed'] for r in records)}",
            flush=True,
        )
    public = {
        "schema_version": 1,
        "status": "complete",
        "corruption": corruption,
        "severity": severity,
        "configuration": "adaptive ProtoTTA; continuous stream",
        "num_records": len(records),
        "num_disagreements": sum(record["changed"] for record in records),
        "records": records,
    }
    reject_label_fields(public)
    sample_ids = [record["sample_id"] for record in records]
    sealed = {
        "schema_version": 1,
        "ordered_sample_ids": sample_ids,
        "targets": targets,
        "target_hash_sha256": sha256_lines(targets),
    }
    atomic_write_json(destination / "public.json", public)
    atomic_write_json(destination / "sealed_targets.json", sealed)
    atomic_write_json(marker, {
        "status": "complete", "labels_excluded_from_public": True,
        "num_records": len(records), "num_disagreements": public["num_disagreements"],
    })
    print(f"export={marker}")
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Datasets" / "Amazon-C")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
