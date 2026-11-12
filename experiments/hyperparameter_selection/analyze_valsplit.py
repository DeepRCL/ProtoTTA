#!/usr/bin/env python
"""Hyperparameter re-selection on the 20% validation split (Table 12, last column).

  select  runs of make_commands.py valsplit-select (validation photos, seed 0): per stream and method,
          the configuration with the highest mean accuracy over the 13 corruptions; ties -> lower learning
          rate, then grid order. Writes the selection JSON read by make_commands.py valsplit-test.
  test    runs of make_commands.py valsplit-test (the other 80% of the photos, seeds 0, 2, 3): selected
          ProtoTTA - selected EATA, mean over the 13 corruptions, averaged over seeds, per stream.

  python analyze_valsplit.py select --results /path/to/val --out selection.json
  python analyze_valsplit.py test --results /path/to/test --selection selection.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import SEEDS  # noqa: E402
from grid import STREAMS, grid  # noqa: E402

VAL_SEED = 0


def read(path):
    (_, res), = json.loads(Path(path).read_text())["results"].items()
    out = {c: v["5"]["accuracy"] for c, v in res.items()}
    if len(out) != 13:
        raise SystemExit(f"{path}: {len(out)} corruptions")
    return out


def mean13(r):
    return 100 * float(np.mean(list(r.values())))


def run_select(root, out_path):
    cands = [{"method": m, "config": c, "knobs": k, "args": list(a)} for m, c, k, a in grid()]
    sel, scores = {}, {}
    for st in STREAMS:
        scores[st] = {g["config"]: mean13(read(root / st / f"{g['config']}_seed{VAL_SEED}.json")) for g in cands}
        sel[st] = {}
        for method in ("prototta", "eata"):
            mc = [(i, g) for i, g in enumerate(cands) if g["method"] == method]
            i, g = min(mc, key=lambda ig: (-round(scores[st][ig[1]["config"]], 9),
                                           float(ig[1]["knobs"]["lr"]), ig[0]))
            sel[st][method] = {"config": g["config"], "knobs": g["knobs"], "args": g["args"],
                               "val_score": scores[st][g["config"]]}
            print(f"{st} {method}: selected {g['config']} (validation {scores[st][g['config']]:.2f})")
    Path(out_path).write_text(json.dumps({"val_seed": VAL_SEED, "selection": sel, "val_scores": scores}, indent=1))
    print(f"wrote {out_path}")


def run_test(root, sel_path):
    sel = json.loads(Path(sel_path).read_text())["selection"]
    diffs = {}
    print("Selected ProtoTTA - selected EATA on the test photos (points; mean over 13 corruptions, seeds 0, 2, 3)")
    for st in STREAMS:
        per_seed = {m: [mean13(read(root / st / f"{sel[st][m]['config']}_seed{s}.json")) for s in SEEDS]
                    for m in ("prototta", "eata")}
        d = [x - y for x, y in zip(per_seed["prototta"], per_seed["eata"])]
        diffs[st] = float(np.mean(d))
        print(f"| {st} | {diffs[st]:+.2f} | (ProtoTTA {sel[st]['prototta']['config']}, EATA {sel[st]['eata']['config']})")
    print(f"| Average | {float(np.mean(list(diffs.values()))):+.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "test"])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="select: selection JSON to write")
    ap.add_argument("--selection", default=None, help="test: selection JSON")
    a = ap.parse_args()
    if a.stage == "select":
        if not a.out:
            raise SystemExit("select needs --out")
        run_select(Path(a.results), a.out)
    else:
        if not a.selection:
            raise SystemExit("test needs --selection")
        run_test(Path(a.results), a.selection)


if __name__ == "__main__":
    main()
