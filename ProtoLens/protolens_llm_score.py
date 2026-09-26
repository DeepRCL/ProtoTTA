#!/usr/bin/env python3
"""Text-only, label-free Qwen supervisor for exported ProtoLens streams."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from protolens_llm_common import (
    MODEL_ID, atomic_write_json, build_prompt, load_json, parse_decisions,
    reject_label_fields, stream_dir, task_for_index,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "llm_prototta"
MODES = ("output_only", "prototype_evidence")
SCORE_FILES = {
    "output_only": "scores_output.json",
    "prototype_evidence": "scores_prototype.json",
}


def verify_hardware(minimum_vram_gib: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("LLM scoring requires CUDA")
    props = torch.cuda.get_device_properties(0)
    memory = props.total_memory / (1024 ** 3)
    name = props.name
    if minimum_vram_gib > 0 and memory < minimum_vram_gib:
        raise RuntimeError(f"{name} has {memory:.1f} GiB; require >= {minimum_vram_gib:g}")
    return {
        "name": name,
        "total_memory_gib": memory,
        "device_count": torch.cuda.device_count(),
    }


def load_model(model_name: str):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return processor, model


def generate_response(
    processor: Any, model: Any, records: Sequence[dict[str, Any]],
    mode: str, max_new_tokens: int,
) -> str:
    prompt = build_prompt(records, mode)
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
        )
    prefix = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        [generated[0, prefix:]], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def score_with_recovery(
    records: Sequence[dict[str, Any]], mode: str,
    generate: Callable[[Sequence[dict[str, Any]], str], str],
    on_valid: Callable[[Sequence[dict[str, Any]], list[dict[str, Any]], str], None],
    retries: int = 5,
) -> None:
    ids = [f"S{index:02d}" for index in range(1, len(records) + 1)]
    raw = generate(records, mode)
    try:
        decisions = parse_decisions(raw, ids)
    except ValueError as exc:
        if len(records) > 1:
            midpoint = len(records) // 2
            score_with_recovery(records[:midpoint], mode, generate, on_valid, retries)
            score_with_recovery(records[midpoint:], mode, generate, on_valid, retries)
            return
        last_error = exc
        for _ in range(retries - 1):
            raw = generate(records, mode)
            try:
                decisions = parse_decisions(raw, ["S01"])
                break
            except ValueError as retry_error:
                last_error = retry_error
        else:
            raise RuntimeError(
                f"malformed singleton response for {records[0]['sample_id']}"
            ) from last_error
    on_valid(records, decisions, raw)


def score_mode(
    records: list[dict[str, Any]], mode: str, destination: Path,
    processor: Any, model: Any, args: argparse.Namespace, hardware: dict[str, Any],
) -> None:
    if destination.is_file():
        progress = load_json(destination)
        if progress.get("status") == "complete":
            return
    else:
        progress = {
            "schema_version": 1,
            "status": "running",
            "mode": mode,
            "model": args.model_name,
            "ground_truth_visible_to_llm": False,
            "configuration": {
                "enable_thinking": False, "torch_dtype": "torch.bfloat16",
                "attn_implementation": "sdpa", "device_map": "auto",
                "do_sample": False, "max_new_tokens": args.max_new_tokens,
                "batch_size": args.llm_batch_size,
            },
            "hardware": hardware,
            "decisions": {},
            "raw_generations": [],
            "valid_generation_batches": 0,
        }
    decisions = progress["decisions"]
    raw_generations = progress.setdefault("raw_generations", [])

    def generate(batch: Sequence[dict[str, Any]], condition: str) -> str:
        return generate_response(
            processor, model, batch, condition, args.max_new_tokens
        )

    def save_valid(
        batch: Sequence[dict[str, Any]], parsed: list[dict[str, Any]], raw: str,
    ) -> None:
        generation_index = len(raw_generations)
        raw_generations.append({
            "sample_ids": [record["sample_id"] for record in batch],
            "response": raw,
        })
        for record, decision in zip(batch, parsed):
            decisions[record["sample_id"]] = {
                "sample_id": record["sample_id"],
                "stream_index": record["stream_index"],
                "action": decision["action"],
                "adaptation_score": decision["adaptation_score"],
                "rationale": decision["rationale"],
                "raw_generation_index": generation_index,
            }
        progress["valid_generation_batches"] += 1
        progress["completed_decisions"] = len(decisions)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(destination, progress)
        print(
            f"VALID mode={mode} completed={len(decisions)}/{len(records)}",
            flush=True,
        )

    for start in range(0, len(records), args.llm_batch_size):
        original = records[start:start + args.llm_batch_size]
        pending = [r for r in original if r["sample_id"] not in decisions]
        if pending:
            score_with_recovery(pending, mode, generate, save_valid)
    expected = {record["sample_id"] for record in records}
    if set(decisions) != expected:
        raise RuntimeError(f"{mode} scores do not cover all disagreements")
    progress["status"] = "complete"
    progress["completed_at"] = datetime.now(timezone.utc).isoformat()
    progress["completed_decisions"] = len(decisions)
    atomic_write_json(destination, progress)


def run(args: argparse.Namespace) -> Path:
    corruption, severity = task_for_index(args.task_id)
    directory = stream_dir(args.output_dir.resolve(), corruption, severity)
    public_path = directory / "public.json"
    marker_path = directory / "export.complete.json"
    if not public_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"export incomplete: {directory}")
    public = load_json(public_path)
    reject_label_fields(public)
    if public.get("status") != "complete":
        raise RuntimeError(f"public export incomplete: {directory}")
    records = [record for record in public["records"] if record["changed"]]
    for record in records:
        for required in (
            "corrupted_review", "before_evidence", "after_evidence",
            "before_prediction", "after_prediction", "before_msp", "after_msp",
        ):
            if required not in record:
                raise RuntimeError(f"missing {required}: {record['sample_id']}")
    hardware = verify_hardware(args.minimum_vram_gib)
    processor, model = load_model(args.model_name)
    for mode in MODES:
        score_mode(
            records, mode, directory / SCORE_FILES[mode],
            processor, model, args, hardware,
        )
    marker = directory / "llm.complete.json"
    atomic_write_json(marker, {
        "status": "complete", "labels_loaded": False,
        "model": args.model_name, "text_only": True,
        "num_queried": len(records),
        "output_scores": str(directory / SCORE_FILES["output_only"]),
        "prototype_scores": str(directory / SCORE_FILES["prototype_evidence"]),
    })
    print(json.dumps({"status": "complete", "artifact": str(marker)}))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--model-name", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--llm-batch-size", type=int, default=6)
    parser.add_argument("--minimum-vram-gib", type=float, default=0.0)
    args = parser.parse_args()
    if args.llm_batch_size < 1:
        parser.error("--llm-batch-size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
