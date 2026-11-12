#!/usr/bin/env python
"""Hyperparameter re-selection over the corruption families (Table 12, first two columns).

Reads the runs of make_commands.py families (<results>/<stream>/<config>_seed<s>.json). For every seed s
and method (ProtoTTA, EATA), configurations are scored by their mean accuracy over the selection
corruptions of seed s; among tied configurations the lower learning rate (then the earlier grid entry)
is chosen.
  Select on three families, test on one: for each family F, select on the other three families and
      record the accuracy of the chosen configuration on the corruptions of F.
  Select on one family, test on three: for each family F, select on F and record the accuracy on the
      other three families; each corruption is tested in three folds and its accuracy is the mean over
      them.
The seed score is the equal-weight mean over the 13 corruptions; the table reports ProtoTTA - EATA,
averaged over seeds, for each stream and the average over the three streams. Exact ties in the
one-family selection are listed with the result of the other tied configuration.

  python analyze_families.py --results /path/to/families
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import FAMILIES, SEEDS  # noqa: E402
from grid import STREAMS, grid, ordered  # noqa: E402

CORR = [c for cs in FAMILIES.values() for c in cs]
assert len(CORR) == len(set(CORR)) == 13
METHODS = ("prototta", "eata")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def load(root):
    acc, missing = {}, []
    for _, c, _, _ in grid():
        for s in SEEDS:
            f = root / f"{c}_seed{s}.json"
            if not f.exists():
                missing.append(f.name)
                continue
            (mode, per_corr), = json.loads(f.read_text())["results"].items()
            a = {k: per_corr[k]["5"]["accuracy"] for k in per_corr}
            if sorted(a) != sorted(CORR):
                raise ValueError(f"{f}: corruptions {sorted(a)}")
            acc[(c, s)] = a
    return acc, missing


def select(configs, seeds, acc, train):
    """Config with the highest equal-weight mean over `train` corruptions (pooled over `seeds`).
    `configs` is in tie-break order; the first of equal scores is kept."""
    def score(c):
        return mean(acc[(c, s)][k] for s in seeds for k in train)
    best = configs[0]
    for c in configs[1:]:
        if score(c) > score(best):
            best = c
    return best


def select_three(configs, s, acc):
    """Per-corruption accuracy when each family is tested after selecting on the other three."""
    out = {}
    for fam, test in FAMILIES.items():
        c = select(configs, [s], acc, [k for k in CORR if k not in test])
        for k in test:
            out[k] = acc[(c, s)][k]
    return out


def select_one(configs, s, acc):
    """Per-corruption accuracy (mean over the 3 folds that test it) and the pick per selection family."""
    tested = {k: [] for k in CORR}
    chosen = {}
    for fam, sel in FAMILIES.items():
        c = select(configs, [s], acc, sel)
        chosen[fam] = c
        for k in CORR:
            if k not in sel:
                tested[k].append(acc[(c, s)][k])
    assert all(len(v) == 3 for v in tested.values())
    return {k: mean(v) for k, v in tested.items()}, chosen


def ties_one(acc, by):
    rows = []
    for m in METHODS:
        for s in SEEDS:
            for fam, sel in FAMILIES.items():
                sc = {c: mean(acc[(c, s)][k] for k in sel) for c in by[m]}
                tied = [c for c in by[m] if sc[c] == max(sc.values())]
                if len(tied) > 1:
                    rows.append({"method": m, "seed": s, "select_on": fam, "tied": tied})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    by = {m: ordered(m) for m in METHODS}
    out = {}
    for st in STREAMS:
        acc, missing = load(Path(a.results) / st)
        if missing:
            raise SystemExit(f"{st}: {len(missing)} results missing (e.g. {missing[:3]})")
        three = {m: {s: select_three(by[m], s, acc) for s in SEEDS} for m in METHODS}
        one = {m: {s: select_one(by[m], s, acc) for s in SEEDS} for m in METHODS}
        d3 = [mean(three["prototta"][s][k] - three["eata"][s][k] for k in CORR) for s in SEEDS]
        d1 = [mean(one["prototta"][s][0].values()) - mean(one["eata"][s][0].values()) for s in SEEDS]
        res = {"select3_test1": 100 * mean(d3), "select3_test1_per_seed": [100 * x for x in d3],
               "select1_test3": 100 * mean(d1), "select1_test3_per_seed": [100 * x for x in d1],
               "ties_select1": []}
        for t in ties_one(acc, by):
            m, s, fam = t["method"], t["seed"], t["select_on"]
            c0, c1 = t["tied"][0], t["tied"][1]
            rest = [k for k in CORR if k not in FAMILIES[fam]]
            delta = sum(acc[(c1, s)][k] - acc[(c0, s)][k] for k in rest) / 3 / len(CORR)
            sign = 1 if m == "prototta" else -1
            t["select1_test3_with_other_tied_config"] = 100 * (mean(d1) + sign * delta / len(SEEDS))
            res["ties_select1"].append(t)
        out[st] = res
    print("ProtoTTA - EATA (points; mean over 13 corruptions and seeds 0, 2, 3)")
    print("| Test stream | Select on 3 families, test on 1 | Select on 1 family, test on 3 |\n|---|---|---|")
    for st, r in out.items():
        print(f"| {st} | {r['select3_test1']:+.2f} | {r['select1_test3']:+.2f} |")
    print(f"| Average | {mean(r['select3_test1'] for r in out.values()):+.2f} | "
          f"{mean(r['select1_test3'] for r in out.values()):+.2f} |")
    for st, r in out.items():
        for t in r["ties_select1"]:
            print(f"tie ({st}, {t['method']}, seed {t['seed']}, select on {t['select_on']}): {t['tied']}; "
                  f"chosen {t['tied'][0]}; with {t['tied'][1]} the {st} select-1 entry is "
                  f"{t['select1_test3_with_other_tied_config']:+.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
