#!/usr/bin/env python3
"""Standalone VLM prompt debugger for a saved ProtoViT sample folder."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from vlm_eval import VLMScorer, ensure_dir, find_json_candidate, load_json, write_json


DEFAULT_SAMPLE_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "vlm_eval"
    / "prototta"
    / "samples"
    / "sample_253_gaussian_noise"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt experiments on one saved VLM evaluation sample."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Saved sample folder containing 00/01/02 images and 03_meta.json.",
    )
    parser.add_argument(
        "--vlm-model-id",
        type=str,
        default="Qwen/Qwen3-VL-32B-Thinking",
        help="Hugging Face VLM model id.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2056,
        help="Maximum generation length.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["original", "strict_json", "final_json", "tagged_json", "all"],
        default="all",
        help="Prompt variant to test.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def image_paths(sample_dir: Path) -> List[Path]:
    return [
        sample_dir / "00_corrupted_input.png",
        sample_dir / "01_predicted_class_reasoning.png",
        sample_dir / "02_any_class_reasoning.png",
    ]


def build_base_prompt(meta: Dict) -> str:
    return (
        "You are evaluating the quality of a prototype-based bird classifier's reasoning.\n"
        "The model classifies birds using 'this looks like that' logic - it matches regions "
        "of the input image to learned prototype patches from training.\n\n"
        "You will receive THREE IMAGES in this exact order:\n"
        "IMAGE 1 = the raw corrupted test image.\n"
        "IMAGE 2 = the predicted-class reasoning board.\n"
        "  - Panel A = raw test image.\n"
        "  - Panel B = same image with a heatmap of the strongest matched prototype.\n"
        "  - Panels C1-C5 = TRAINING prototype patches retrieved by the model.\n"
        "  - These are NOT crops from the test image.\n"
        "  - 'Pxxx' is the prototype id and the last number under each patch is its contribution score.\n"
        "IMAGE 3 = the any-class reasoning board.\n"
        "  - Same layout as IMAGE 2, but Panels C1-C10 are strongest prototypes contributing to ANY class.\n\n"
        f"The model predicted: {meta['predicted_class']}\n"
        f"The correct answer is: {meta['ground_truth_class']}\n"
        f"This prediction is: {'CORRECT' if meta['is_correct'] else 'WRONG'}\n\n"
        "Judge the model mostly from IMAGE 2. Use IMAGE 1 for the corrupted bird appearance, "
        "and IMAGE 3 to detect spurious wrong-class evidence.\n\n"
        "Return scores on these dimensions:\n"
        "part_coherence_score: integer 1-5\n"
        "prototype_match_score: integer 1-5\n"
        "overall_adaptation_quality: integer 1-5\n"
        "part_name: one word bird part\n"
        "one_sentence_summary: one sentence\n"
    )


def prompt_variants(meta: Dict) -> Dict[str, str]:
    base = build_base_prompt(meta)
    return {
        "original": (
            base
            + "\nRespond in JSON only with keys: part_coherence_score, prototype_match_score, "
            "overall_adaptation_quality, part_name, one_sentence_summary."
        ),
        "strict_json": (
            base
            + "\nOutput exactly one JSON object. Do not explain. Do not think aloud. "
            "Do not use markdown fences. The first character must be '{' and the last character must be '}'."
        ),
        "final_json": (
            base
            + "\nThink silently. Then output only the final answer as exactly one JSON object. "
            "No reasoning, no analysis, no markdown."
        ),
        "tagged_json": (
            base
            + "\nDo any internal reasoning silently. Your visible output must be only:\n"
            "<final_json>\n"
            "{\"part_coherence_score\": 1, \"prototype_match_score\": 1, \"overall_adaptation_quality\": 1, "
            "\"part_name\": \"wing\", \"one_sentence_summary\": \"example\"}\n"
            "</final_json>\n"
            "Replace the example values with the real answer."
        ),
    }


def collect_candidates(raw_text: str) -> Dict[str, str]:
    candidate = find_json_candidate(raw_text)
    tagged = ""
    start_tag = "<final_json>"
    end_tag = "</final_json>"
    if start_tag in raw_text and end_tag in raw_text:
        tagged = raw_text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return {
        "find_json_candidate": candidate,
        "tagged_candidate": tagged,
    }


def try_parse_candidates(candidates: Dict[str, str]) -> Dict:
    for _, candidate in candidates.items():
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("No parseable JSON candidate found in raw VLM output.")


def run_prompt_mode(
    sample_dir: Path,
    mode: str,
    prompt: str,
    scorer: VLMScorer,
) -> None:
    logging.info("Running prompt mode '%s' for %s", mode, sample_dir.name)
    debug_dir = sample_dir / "prompt_debug" / mode
    ensure_dir(debug_dir)
    write_json(debug_dir / "run_info.json", {"mode": mode, "sample_dir": str(sample_dir)})
    (debug_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    raw_text = ""
    try:
        logging.info("Sending prompt mode '%s' to VLM", mode)
        _, raw_text, used_prompt = scorer.generate_json(image_paths(sample_dir), prompt)
        logging.info("Received VLM output for prompt mode '%s' (%d chars)", mode, len(raw_text))
        (debug_dir / "used_prompt.txt").write_text(used_prompt, encoding="utf-8")
        (debug_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
        candidates = collect_candidates(raw_text)
        write_json(debug_dir / "candidates.json", candidates)
        parsed = try_parse_candidates(candidates)
        write_json(debug_dir / "parsed.json", parsed)
        logging.info("Successfully parsed JSON for prompt mode '%s'", mode)
        write_json(
            debug_dir / "summary.json",
            {"status": "parsed", "mode": mode, "parsed_path": str(debug_dir / "parsed.json")},
        )
    except Exception as exc:
        logging.error("Prompt mode '%s' failed: %s", mode, exc)
        raw_text = getattr(exc, "raw_text", raw_text)
        if raw_text:
            (debug_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
            write_json(debug_dir / "candidates.json", collect_candidates(raw_text))
        used_prompt = getattr(exc, "prompt_text", prompt)
        (debug_dir / "used_prompt.txt").write_text(used_prompt, encoding="utf-8")
        (debug_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        (debug_dir / "error_repr.txt").write_text(repr(exc), encoding="utf-8")
        write_json(debug_dir / "summary.json", {"status": "error", "mode": mode, "error": str(exc)})


def main() -> None:
    setup_logging()
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    meta = load_json(sample_dir / "03_meta.json")
    variants = prompt_variants(meta)
    modes = list(variants.keys()) if args.prompt_mode == "all" else [args.prompt_mode]

    scorer = VLMScorer(args.vlm_model_id, args.max_new_tokens)
    for mode in modes:
        run_prompt_mode(sample_dir, mode, variants[mode], scorer)


if __name__ == "__main__":
    main()
