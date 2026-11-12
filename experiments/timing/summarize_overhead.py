#!/usr/bin/env python
"""Per-image overhead of the full controller (Section 5, Appendix N; runs of make_commands.py overhead).

Per backbone and method: time per sample = mean over the three corruptions of the per-corruption
median over the repetitions of efficiency.time_per_sample_ms. Overhead = ProtoTTA (adaptive lambda)
minus unadapted, in ms per image.

  python summarize_overhead.py --results /path/to/timing_overhead
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import times  # noqa: E402


def ms(files):
    reps = [times(f) for f in files]
    if not reps or any(len(r) != 3 or sorted(r) != sorted(reps[0]) for r in reps):
        raise SystemExit(f"expected the same 3 settings in every repetition: {files}")
    med = {k: statistics.median(r[k] for r in reps) for k in reps[0]}
    return sum(med.values()) / len(med)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    print("| Backbone | unadapted ms/image | ProtoTTA ms/image | overhead ms/image |\n|---|---|---|---|")
    for b in ("protovit", "protopformer", "protopnet"):
        d = Path(a.results) / b
        if not d.exists():
            continue
        u = ms(sorted(d.glob("unadapted_rep*.json")))
        p = ms(sorted(d.glob("adaptive_rep*.json")))
        print(f"| {b} | {u:.2f} | {p:.2f} | {p - u:.2f} |")


if __name__ == "__main__":
    main()
