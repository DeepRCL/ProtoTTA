#!/usr/bin/env python
"""Table 3: objective x reliability filter (left, ProtoViT) and controller components (right, four backbones).

Accuracy of a cell = mean over the corruption settings of the seed-averaged per-setting accuracy
(seeds 0, 2, 3). Cells are read from
  --results DIR                       runs written by make_commands.py (<backbone>/<variant>_seed<s>.json,
                                      fixed-lambda runs as fixed_<lambda>_seed<s>.json);
  --sweep BACKBONE:LAMBDA FILE ...    fixed-lambda runs given explicitly (e.g. the Table 10 sweep), one file
                                      per seed;
  --row BACKBONE:VARIANT FILE ...     any other runs of a cell (e.g. the Table 1 run of the router),
the later sources overriding the earlier ones. The fixed-lambda rows are the lambda = 0, 0.5 and 1 runs;
"Best fixed lambda" is the best fixed lambda of the backbone. The largest-shortfall column is, for each
row, the largest difference to the best row of the same backbone, computed on the values as displayed
(2 decimals).

  python analyze.py --results /path/to/results \
      --sweep protovit:0.0 sweep_pv/lambda0.0_seed{0,2,3}.json ... --row protopnet:router table1_ppn/*.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SEEDS, seed_averaged  # noqa: E402

BACKBONES = ("protovit", "protopnet", "protopformer", "protolens")
N_SETTINGS = {"protovit": 13, "protopformer": 13, "protopnet": 13, "protolens": 20}
RIGHT = [("router", "Router (ours)"), ("act_only", "lambda^act only"), ("rel_only", "lambda^rel only"),
         ("mean", "Mean of the two"), ("no_coverage_guard", "No coverage guard"),
         ("no_grad_norm", "No gradient norm."), ("fixed_0", "Fixed lambda = 0 (output)"),
         ("fixed_0.5", "Fixed lambda = 0.5"), ("fixed_1", "Fixed lambda = 1 (prototype)")]
FIXED = {"fixed_0": 0.0, "fixed_0.5": 0.5, "fixed_1": 1.0}
LEFT = {("Output entropy", "None"): "output_nofilter", ("Output entropy", "EATA rule"): "output_entropyfilter",
        ("Output entropy", "Ours"): "fixed_0", ("ProtoTTA (adaptive)", "None"): "router_nofilter",
        ("ProtoTTA (adaptive)", "EATA rule"): "router_entropyfilter", ("ProtoTTA (adaptive)", "Ours"): "router"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--sweep", nargs="+", action="append", default=[], metavar=("BACKBONE:LAMBDA", "FILE"))
    ap.add_argument("--row", nargs="+", action="append", default=[], metavar=("BACKBONE:VARIANT", "FILE"))
    a = ap.parse_args()
    cells, source = {}, {}

    def put(b, v, files, what):
        r = seed_averaged(files, N_SETTINGS[b])
        if r is not None:
            cells[(b, v)] = r[0]
            source[(b, v)] = what

    variants = [v for v, _ in RIGHT if v not in FIXED] + [v for v in LEFT.values() if v not in dict(RIGHT)]
    sweep = {}
    if a.results:
        for b in BACKBONES:
            for v in variants:
                put(b, v, [Path(a.results) / b / f"{v}_seed{s}.json" for s in a.seeds], "results")
            lams = sorted({float(f.name[len("fixed_"):].rsplit("_seed", 1)[0])
                           for f in (Path(a.results) / b).glob("fixed_*_seed*.json")})
            for lam in lams:
                r = seed_averaged([Path(a.results) / b / f"fixed_{lam:g}_seed{s}.json" for s in a.seeds],
                                  N_SETTINGS[b])
                if r is not None:
                    sweep.setdefault(b, {})[lam] = (r[0], "results")
    for spec in a.sweep:
        b, lam = spec[0].split(":")
        r = seed_averaged(spec[1:], N_SETTINGS[b])
        if r is None:
            raise SystemExit(f"--sweep {spec[0]}: file missing")
        sweep.setdefault(b, {})[float(lam)] = (r[0], "sweep " + " ".join(spec[1:]))
    for b, d in sweep.items():
        for v, x in FIXED.items():
            if x in d:
                cells[(b, v)], source[(b, v)] = d[x]
    for spec in a.row:
        b, v = spec[0].split(":")
        r = seed_averaged(spec[1:], N_SETTINGS[b])
        if r is None:
            raise SystemExit(f"--row {spec[0]}: file missing")
        cells[(b, v)] = r[0]
        source[(b, v)] = "row " + " ".join(spec[1:])

    shown = {k: round(x, 2) for k, x in cells.items()}
    print("## Table 3 (left): objective x reliability filter, ProtoViT (mean accuracy %)")
    print("| Objective \\ Filter | None | EATA rule | Ours |\n|---|---|---|---|")
    for obj in ("Output entropy", "ProtoTTA (adaptive)"):
        row = [shown.get(("protovit", LEFT[(obj, f)])) for f in ("None", "EATA rule", "Ours")]
        print(f"| {obj} | " + " | ".join("-" if x is None else f"{x:.2f}" for x in row) + " |")

    rows = [(v, label) for v, label in RIGHT] + [("best_fixed", "Best fixed lambda (sweep using labels)")]
    for b, d in sweep.items():
        lam = max(d, key=lambda x: d[x][0])
        shown[(b, "best_fixed")] = round(d[lam][0], 2)
        source[(b, "best_fixed")] = f"lambda={lam:g} of {sorted(d)}"
    best = {}
    for b in BACKBONES:
        vals = [shown[(b, v)] for v, _ in rows if (b, v) in shown]
        if vals:
            best[b] = max(vals)
    print("\n## Table 3 (right): controller components (mean accuracy %)")
    print("| Variant | ProtoViT | ProtoPNet | ProtoPFormer | ProtoLens | Largest shortfall |\n|---|---|---|---|---|---|")
    for v, label in rows:
        vals = [shown.get((b, v)) for b in BACKBONES]
        gaps = [best[b] - shown[(b, v)] for b in BACKBONES if (b, v) in shown and b in best]
        cols = " | ".join("-" if x is None else f"{x:.2f}" for x in vals)
        short = f"{max(gaps):.2f}" if len(gaps) == len(BACKBONES) else "-"
        print(f"| {label} | {cols} | {short} |")
    print("\nCell sources:")
    for (b, v), s in sorted(source.items()):
        print(f"  {b}:{v}: {s}")


if __name__ == "__main__":
    main()
