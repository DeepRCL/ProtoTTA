#!/usr/bin/env python
"""104-pair subset of the narrator validation set (the set narrated in Table 5, right).

Keeps, for every (stratum, corruption) cell, the first PER_CELL pairs in draw order (pairs are
stored cell by cell in draw order by narrator_items.py; the draw order is seeded random), and the
plan entries of those pairs in their original (seeded, shuffled) order. Ground truth, boards and
the plan are otherwise unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PV = HERE.parent
PER_CELL = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(PV / "results/explanation_eval/narrator/items.json"))
    ap.add_argument("--out", default=str(PV / "results/explanation_eval/narrator/items_104.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    raw = Path(args.items).read_bytes()
    doc = json.loads(raw)
    assert not any(p["filled_from_other_corruption"] for p in doc["pairs"]) and not doc["shortfall"]
    seen, keep = {}, []
    for p in doc["pairs"]:
        cell = (p["stratum"], p["corruption"])
        seen[cell] = seen.get(cell, 0) + 1
        if seen[cell] <= PER_CELL:
            keep.append(p)
    assert all(v == doc["per_cell_quota"] for v in seen.values()), seen
    ids = {p["pair_id"] for p in keep}
    plan = [s for s in doc["plan"] if s["pair_id"] in ids]
    assert len(keep) == PER_CELL * len(seen) and len(plan) == 3 * len(keep)
    sub = {**doc, "pairs": keep, "plan": plan, "per_cell_quota": PER_CELL,
           "subset_of": {"items_sha256": hashlib.sha256(raw).hexdigest(), "rule": __doc__,
                         "code_sha256": hashlib.sha256((HERE / "narrator_subset.py").read_bytes()).hexdigest()}}
    out.write_text(json.dumps(sub))
    by = {}
    for p in keep:
        by[p["stratum"]] = by.get(p["stratum"], 0) + 1
    print(f"{len(keep)} pairs {by}, {len(plan)} presentations -> {out}")


if __name__ == "__main__":
    main()
