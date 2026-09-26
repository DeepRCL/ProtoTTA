"""Score label-free ProtoTTA disagreements with Qwen3.6.

This module intentionally knows only ``public.json``.  It recursively splits
malformed batches, validates every ID/action/score, and atomically checkpoints
after every valid generation so interrupted jobs resume without re-scoring.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from .vlm_prototta_common import (
    CORRUPTIONS,
    VLM_BATCH_SIZE,
    VLM_MODEL,
    atomic_write_json,
    build_prompt,
    load_json,
    parse_and_validate_response,
    reject_label_bearing_payload,
)


def verify_hardware(minimum_vram_gib: float = 0.0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen scoring requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / 1024**3
    name = properties.name
    if minimum_vram_gib > 0 and memory_gib < minimum_vram_gib:
        raise RuntimeError(
            f"{name} has {memory_gib:.1f} GiB; require >={minimum_vram_gib:g} GiB"
        )
    return {
        "name": name,
        "total_memory_gib": memory_gib,
        "device_count": torch.cuda.device_count(),
    }


def load_qwen(model_name: str, torch_dtype: str = "bfloat16"):
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "latest Hugging Face transformers with AutoModelForMultimodalLM "
            "is required for Qwen VLM scoring"
        ) from exc
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[torch_dtype]
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return processor, model


def _image_path(record: dict[str, Any], mode: str) -> str:
    key = "image_path" if mode == "image_only" else "board_path"
    value = record.get(key)
    if not value or not Path(value).is_file():
        raise FileNotFoundError(f"missing {mode} image for {record['sample_id']}: {value}")
    return str(Path(value).resolve())


def generate_response(
    processor: Any,
    model: Any,
    records: Sequence[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(records, mode)
    content = [
        {"type": "image", "image": _image_path(record, mode)}
        for record in records
    ]
    content.append({"type": "text", "text": prompt})
    conversation = [{"role": "user", "content": content}]
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
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    prefix_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        [generated[0, prefix_length:]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def score_with_splitting(
    records: Sequence[dict[str, Any]],
    mode: str,
    generate: Callable[[Sequence[dict[str, Any]], str], str],
    on_valid: Callable[[Sequence[dict[str, Any]], list[dict], str], None],
    singleton_retries: int = 5,
) -> None:
    """Generate one multi-image response, recursively splitting on failure."""
    expected_ids = [f"S{index:02d}" for index in range(1, len(records) + 1)]
    raw = generate(records, mode)
    try:
        decisions = parse_and_validate_response(raw, expected_ids)
    except ValueError as exc:
        if len(records) > 1:
            midpoint = len(records) // 2
            score_with_splitting(
                records[:midpoint], mode, generate, on_valid, singleton_retries
            )
            score_with_splitting(
                records[midpoint:], mode, generate, on_valid, singleton_retries
            )
            return
        last_error = exc
        for _ in range(singleton_retries - 1):
            try:
                raw = generate(records, mode)
                decisions = parse_and_validate_response(raw, ["S01"])
                break
            except ValueError as retry_error:
                last_error = retry_error
        else:
            raise RuntimeError(
                f"malformed singleton response for {records[0]['sample_id']}"
            ) from last_error
    on_valid(records, decisions, raw)


def score_mode(
    records: list[dict[str, Any]],
    mode: str,
    output_path: Path,
    processor: Any,
    model: Any,
    model_name: str,
    hardware: dict[str, Any],
    max_new_tokens: int,
    torch_dtype: str = "bfloat16",
    vlm_batch_size: int = VLM_BATCH_SIZE,
) -> None:
    if output_path.is_file():
        progress = load_json(output_path)
        if progress.get("status") == "complete":
            return
    else:
        progress = {
            "schema_version": 1,
            "status": "running",
            "mode": mode,
            "model": model_name,
            "configuration": {
                "enable_thinking": False,
                "torch_dtype": f"torch.{torch_dtype}",
                "attn_implementation": "sdpa",
                "device_map": "auto",
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "vlm_batch_size": vlm_batch_size,
            },
            "hardware": hardware,
            "decisions": {},
            "raw_generations": [],
            "valid_generation_batches": 0,
        }
    decisions_by_id = progress["decisions"]
    raw_generations = progress.setdefault("raw_generations", [])
    first_batch_logged = progress["valid_generation_batches"] > 0

    def generate(batch: Sequence[dict[str, Any]], condition: str) -> str:
        return generate_response(
            processor, model, batch, condition, max_new_tokens
        )

    def save_valid(
        batch: Sequence[dict[str, Any]], decisions: list[dict], raw: str
    ) -> None:
        nonlocal first_batch_logged
        generation_index = len(raw_generations)
        raw_generations.append({
            "sample_ids": [record["sample_id"] for record in batch],
            "response": raw,
        })
        for record, decision in zip(batch, decisions):
            decisions_by_id[record["sample_id"]] = {
                "sample_id": record["sample_id"],
                "stream_index": record["stream_index"],
                "action": decision["action"],
                "adaptation_score": decision["adaptation_score"],
                "rationale": decision["rationale"],
                "raw_generation_index": generation_index,
            }
        progress["valid_generation_batches"] += 1
        progress["completed_decisions"] = len(decisions_by_id)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output_path, progress)
        if not first_batch_logged:
            print(
                "FIRST_QWEN_BATCH_VALID "
                f"mode={mode} samples={len(batch)} ids="
                f"{','.join(record['sample_id'] for record in batch)}",
                flush=True,
            )
            first_batch_logged = True

    # Preserve the original fixed batch partition across resumes. If a malformed
    # batch was partly saved before interruption, only its missing members are
    # retried; later members never move into a different context batch.
    for start in range(0, len(records), vlm_batch_size):
        original_batch = records[start:start + vlm_batch_size]
        pending = [
            record for record in original_batch
            if record["sample_id"] not in decisions_by_id
        ]
        if pending:
            score_with_splitting(pending, mode, generate, save_valid)
    expected = {record["sample_id"] for record in records}
    if set(decisions_by_id) != expected:
        raise RuntimeError("scoring output does not cover every disagreement exactly once")
    progress["status"] = "complete"
    progress["completed_at"] = datetime.now(timezone.utc).isoformat()
    progress["completed_decisions"] = len(decisions_by_id)
    atomic_write_json(output_path, progress)


def score_stream(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    hardware: dict[str, Any],
) -> Path:
    seed, corruption = args.seed, args.corruption
    stream_dir = args.output_dir.resolve() / f"seed_{seed}" / corruption
    public_path = stream_dir / "public.json"
    if not public_path.is_file():
        raise FileNotFoundError(f"export dependency is incomplete: {public_path}")
    public = load_json(public_path)
    reject_label_bearing_payload(public)
    if public.get("status") != "complete":
        raise RuntimeError("public export is incomplete")
    if public["seed"] != seed or public["corruption"] != corruption:
        raise RuntimeError("task/export identity mismatch")
    # Relocate absolute paths recorded by the export host.  Basenames are
    # opaque and the copied run directory retains images/ and boards/.
    for record in public["records"]:
        for key, subdirectory in (
            ("image_path", "images"), ("board_path", "boards")
        ):
            value = record.get(key)
            if value:
                local_path = stream_dir / subdirectory / Path(value).name
                if local_path.is_file():
                    record[key] = str(local_path.resolve())
    records = [record for record in public["records"] if record["changed"]]
    # Check only the allowed images now, before loading the 70B-scale checkpoint.
    for record in records:
        _image_path(record, "image_only")
        _image_path(record, "board")
    score_dir = stream_dir
    if args.score_tag:
        score_dir = stream_dir / "vlm_scores" / args.score_tag
    score_mode(
        records, "image_only", score_dir / "scores_image.json",
        processor, model, args.model_name, hardware, args.max_new_tokens,
        args.torch_dtype, args.vlm_batch_size,
    )
    score_mode(
        records, "board", score_dir / "scores_board.json",
        processor, model, args.model_name, hardware, args.max_new_tokens,
        args.torch_dtype, args.vlm_batch_size,
    )
    complete_path = score_dir / "vlm.complete.json"
    atomic_write_json(complete_path, {
        "status": "complete", "seed": seed, "corruption": corruption,
        "model": args.model_name, "score_tag": args.score_tag,
        "num_queried": len(records),
        "image_scores": str(score_dir / "scores_image.json"),
        "board_scores": str(score_dir / "scores_board.json"),
        "labels_loaded": False,
    })
    return complete_path


def run(args: argparse.Namespace) -> Path:
    hardware = verify_hardware(args.minimum_vram_gib)
    processor, model = load_qwen(args.model_name, args.torch_dtype)
    return score_stream(args, processor, model, hardware)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corruption", choices=CORRUPTIONS, required=True)
    parser.add_argument("--model-name", default=VLM_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--minimum-vram-gib", type=float, default=0.0)
    parser.add_argument("--vlm-batch-size", type=int, default=VLM_BATCH_SIZE)
    parser.add_argument("--score-tag", default="")
    args = parser.parse_args(argv)
    if args.vlm_batch_size < 1:
        parser.error("--vlm-batch-size must be positive")
    if args.score_tag and any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.score_tag):
        parser.error("--score-tag may contain only letters, digits, underscores, and hyphens")
    return args


def main(argv: list[str] | None = None) -> None:
    path = run(parse_args(argv))
    print(json.dumps({"status": "complete", "artifact": str(path)}))


if __name__ == "__main__":
    main()
