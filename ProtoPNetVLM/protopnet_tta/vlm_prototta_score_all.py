"""Load one VLM once and score all 39 exported ProtoPNet streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vlm_prototta_common import CORRUPTIONS, SEEDS, VLM_BATCH_SIZE, VLM_MODEL
from .vlm_prototta_score import load_qwen, score_stream, verify_hardware


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=VLM_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--minimum-vram-gib", type=float, default=90.0)
    parser.add_argument("--vlm-batch-size", type=int, default=VLM_BATCH_SIZE)
    parser.add_argument("--score-tag", required=True)
    args = parser.parse_args(argv)
    if args.vlm_batch_size < 1:
        parser.error("--vlm-batch-size must be positive")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.score_tag):
        parser.error("--score-tag may contain only letters, digits, underscores, and hyphens")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hardware = verify_hardware(args.minimum_vram_gib)
    print(json.dumps({"event": "loading_model", "model": args.model_name,
                      "dtype": args.torch_dtype, "hardware": hardware}), flush=True)
    processor, model = load_qwen(args.model_name, args.torch_dtype)
    print("MODEL_LOAD_COMPLETE", flush=True)
    for seed in SEEDS:
        for corruption in CORRUPTIONS:
            args.seed, args.corruption = seed, corruption
            print(f"STREAM_START seed={seed} corruption={corruption}", flush=True)
            artifact = score_stream(args, processor, model, hardware)
            print(f"STREAM_COMPLETE artifact={artifact}", flush=True)
    print("ALL_QWEN_STREAMS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
