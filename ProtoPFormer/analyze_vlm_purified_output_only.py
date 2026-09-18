#!/usr/bin/env python3
"""Aggregate architecture-neutral, output-only VLM supervision."""
from pathlib import Path

import analyze_vlm_purified as analysis

analysis.RESULTS = (
    Path(__file__).resolve().parent
    / "results/vlm_purified/evaluation_output_only"
)

if __name__ == "__main__":
    analysis.main()
