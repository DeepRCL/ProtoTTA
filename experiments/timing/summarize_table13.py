#!/usr/bin/env python
"""Table 13: speed of the ProtoTTA variants relative to EATA on ProtoViT (runs of make_commands.py table13).

Per run: time per sample = mean over the three corruptions of efficiency.time_per_sample_ms. Per
repetition, speed relative to EATA = EATA time / method time (both from the same repetition); the table
reports the mean and SD over the repetitions.

  python summarize_table13.py --results /path/to/timing_table13
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import times  # noqa: E402

ROWS = (("adaptive", "ProtoTTA, adaptive lambda"), ("fixed_1", "ProtoTTA, fixed lambda = 1"),
        ("calibrate_once", "ProtoTTA, calibrate-once"), ("eata", "EATA"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    root = Path(a.results)
    runs = {}
    for f in sorted(root.glob("*_rep*.json")):
        m, rep = f.stem.rsplit("_rep", 1)
        t = times(f)
        if len(t) != 3:
            raise SystemExit(f"{f.name}: expected 3 settings, got {sorted(t)}")
        runs.setdefault(m, {})[int(rep)] = sum(t.values()) / 3
    reps = sorted(runs["eata"])
    print(f"Speed relative to EATA, mean +- SD over {len(reps)} repetitions")
    print("| Method | Speed relative to EATA |\n|---|---|")
    for m, label in ROWS:
        if sorted(runs.get(m, {})) != reps:
            raise SystemExit(f"{m}: repetitions {sorted(runs.get(m, {}))} != {reps}")
        vs = [runs["eata"][r] / runs[m][r] for r in reps]
        sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
        print(f"| {label} | {100 * statistics.mean(vs):.1f}% +- {100 * sd:.1f} |")


if __name__ == "__main__":
    main()
