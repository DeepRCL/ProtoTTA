#!/usr/bin/env python
"""Mean accuracy over the corruption settings of the seed-averaged per-setting accuracy (%), as in the
paper's tables, for a list of per-seed result files of one method.

  python summarize_accuracy.py /path/to/calibrate_once/calibrate_once_seed{0,2,3}.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import seed_averaged  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    r = seed_averaged(a.files)
    if r is None:
        raise SystemExit("a result file is missing")
    m, sd, per = r
    print(f"{len(a.files)} seeds, {len(per)} settings: mean accuracy {m:.2f} (SD across settings {sd:.2f})")


if __name__ == "__main__":
    main()
