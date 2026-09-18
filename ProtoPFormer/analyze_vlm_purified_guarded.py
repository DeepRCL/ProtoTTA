#!/usr/bin/env python3
"""Aggregate the frozen ProtoViT confidence-guard transfer."""
from pathlib import Path

import analyze_vlm_purified as analysis

analysis.RESULTS = Path(__file__).resolve().parent / "results/vlm_purified/evaluation_guarded"

if __name__ == "__main__":
    analysis.main()
