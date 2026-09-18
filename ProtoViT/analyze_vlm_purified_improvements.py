#!/usr/bin/env python3
"""Rank the predeclared label-free semantic-controller variants on seed 0."""
import json
from pathlib import Path

import numpy as np

import vlm_eval
from semantic_prototype_guidance import write_json

ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / "results/semantic_prototta/improvement_screen"
REFERENCE = ROOT / "results/semantic_prototta/purified_main/purified_main_seed0.json"


def mean_accuracy(path):
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    return float(np.mean([results[name]["online_accuracy"] for name in vlm_eval.CORRUPTION_TYPES]))


def main():
    reference = mean_accuracy(REFERENCE)
    rows = []
    for path in sorted(SCREEN.glob("*_seed0.json")):
        value = mean_accuracy(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "variant": path.stem.removesuffix("_seed0"),
            "mean_accuracy": value,
            "delta_vs_completed_vlm_purified": value - reference,
            "semantic_fusion": payload["semantic_fusion"],
            "semantic_logit_blend": payload["semantic_logit_blend"],
            "semantic_contrast_weight": payload["semantic_contrast_weight"],
        })
    rows.sort(key=lambda item: item["mean_accuracy"], reverse=True)
    output = {"schema_version": 1, "screening_seed": 0,
              "completed_vlm_purified_accuracy": reference, "variants": rows}
    write_json(SCREEN / "summary.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
