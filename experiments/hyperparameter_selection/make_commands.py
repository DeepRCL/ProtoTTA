#!/usr/bin/env python
"""Print the ProtoViT commands of the hyperparameter re-selection (Appendix M, Table 12).

  families        every grid configuration (grid.py) in the three test streams on all CUB-200-C photos,
                  seeds 0, 2, 3 -> <results>/<stream>/<config>_seed<s>.json
                  (used by analyze_families.py: select on three families / test on one, and select on one
                  family / test on three)
  valsplit-select every grid configuration in the three streams on the 20% validation photos, seed 0
                  -> <results>/<stream>/<config>_seed0.json (analyze_valsplit.py select)
  valsplit-test   the selected ProtoTTA and EATA configurations (--selection, from analyze_valsplit.py
                  select) on the remaining 80% of the photos, seeds 0, 2, 3
                  -> <results>/<stream>/<config>_seed<s>.json (analyze_valsplit.py test)
The photo split is built by make_valsplit.py. Commands run from the ProtoViT/ directory.

  python make_commands.py families --results /path/to/families > families.sh
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import SEEDS, command  # noqa: E402
from grid import MODEL, STREAMS, evaluator_args, grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["families", "valsplit-select", "valsplit-test"])
    ap.add_argument("--results", required=True, help="output root (made absolute in the printed commands)")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--python", default="python")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data_dir", default=None,
                    help="default: ./datasets/cub200_c/ (families), ./datasets/cub200_c_val/ (valsplit-select), "
                         "./datasets/cub200_c_test/ (valsplit-test)")
    ap.add_argument("--selection", default=None, help="valsplit-test: selection JSON of analyze_valsplit.py select")
    a = ap.parse_args()
    data = a.data_dir or {"families": "./datasets/cub200_c/", "valsplit-select": "./datasets/cub200_c_val/",
                          "valsplit-test": "./datasets/cub200_c_test/"}[a.stage]
    res, workdir = Path(a.results).resolve(), Path(a.repo).resolve() / "ProtoViT"
    configs = {c: args for m, c, k, args in grid()}
    if a.stage == "families":
        jobs = [(st, c, s) for st in STREAMS for c in configs for s in SEEDS]
    elif a.stage == "valsplit-select":
        jobs = [(st, c, 0) for st in STREAMS for c in configs]
    else:
        if not a.selection:
            raise SystemExit("valsplit-test needs --selection")
        sel = json.loads(Path(a.selection).read_text())["selection"]
        jobs = [(st, sel[st][m]["config"], s) for st in STREAMS for m in ("prototta", "eata") for s in SEEDS]
    for st, c, s in jobs:
        args = evaluator_args(data, st, configs[c], s, res / st / f"{c}_seed{s}.json", a.model)
        print(command(workdir, ["evaluate_robustness.py"], args, a.python))


if __name__ == "__main__":
    main()
