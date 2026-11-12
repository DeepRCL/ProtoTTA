#!/usr/bin/env python
"""Selection of the 16 items of the pairwise human study (ProtoTTA vs EATA).

Population: all CUB-200-C severity-5 test pairs (5,794 images x 13 corruptions) whose
two explanation cards can be compared, i.e. (a) the per-sample state check passed for
EATA and ProtoTTA, (b) both cards have 5 prototypes chosen by the predicted-class rule
(no fallback), and (c) the two cards are NOT identical as displayed (same prototypes
with the same active sub-patch locations).
Stratified sampling (stratum quotas reproduce the correctness mix of the eligible test
population):
  * strata = (ProtoTTA correct, EATA correct). Stratum quotas for N_ITEMS items are the
    population shares among eligible pairs, rounded by the largest-remainder method.
  * corruptions: every corruption gets one item; the remaining N_ITEMS - 13 items go
    to corruptions drawn at random without replacement.
  * the corruption slots and stratum slots are paired by a seeded random permutation
    (a random table with both margins fixed; redrawn if a cell is empty), and each slot
    takes a uniformly random eligible pair from its (corruption, stratum) cell.
  * no source bird image is used twice (also not from the excluded earlier items).
For every slot N_BACKUP further candidates from the same cell (distinct birds) are
listed in order; a slot moves to its next candidate only if the screening judge
(Gemini) cannot return a valid answer for the current one (finalize_pairwise.py).
One more pair from a random cell, with a random method, is the tutorial card.
Labels are used only to form strata; they are never shown.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PV = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from select_items import card_signature  # noqa: E402  (same display signature as select_items.py)

SEED = 20261107
N_ITEMS = 16
N_BACKUP = 3
SEVERITY = 5
METHODS = ("eata", "prototta")
STRATA = (("both_right", True, True), ("both_wrong", False, False),
          ("prototta_right_eata_wrong", True, False), ("eata_right_prototta_wrong", False, True))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratum_of(recs):
    p, e = (recs["prototta"]["pred"] == recs["prototta"]["label"]), (recs["eata"]["pred"] == recs["eata"]["label"])
    return next(name for name, sp, se in STRATA if (sp, se) == (p, e))


def eligible(recs):
    if not all(r["state_check"] for r in recs.values()):
        return "state_check"
    if not all(r["top_rule"] == "predicted_class" and len(r["top"]) == 5
               and all(t["cls"] == r["pred"] and t["w"] > 0 for t in r["top"]) for r in recs.values()):
        return "not_predicted_class_rule"
    if card_signature(recs["prototta"]) == card_signature(recs["eata"]):
        return "identical_cards"
    return None


def largest_remainder(shares, n):
    raw = {k: v * n for k, v in shares.items()}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[: n - sum(q.values())]:
        q[k] += 1
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default=str(PV / "results/explanation_eval/states"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_items", default=None,
                    help="earlier items whose birds must not be reused (select_items.py --mode human output)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    root = Path(args.states_root)
    corruptions = sorted(p.stem[:-3] for p in (root / "prototta").glob(f"*_s{SEVERITY}.json"))
    assert len(corruptions) == 13, corruptions
    ver = {}
    for m in METHODS:
        v = json.loads((root / m / "_verification.json").read_text())
        if not v.get("passed") or v.get("severity") != SEVERITY or v.get("gate") != "exact_count_v2":
            raise SystemExit(f"{m} export did not pass verification")
        ver[m] = v

    data, provenance, skip = {}, {}, {}
    cells = {(c, s[0]): [] for c in corruptions for s in STRATA}
    for c in corruptions:
        data[c] = {}
        for m in METHODS:
            f = root / m / f"{c}_s{SEVERITY}.json"
            digest = file_sha256(f)
            if ver[m]["export_sha256"].get(f.name) != digest:
                raise SystemExit(f"{m}/{f.name} changed after verification")
            provenance[f"{m}/{f.name}"] = digest
            data[c][m] = json.loads(f.read_text())
        assert [s["path"] for s in data[c]["eata"]["samples"]] == [s["path"] for s in data[c]["prototta"]["samples"]]
        skip[c] = {"state_check": 0, "not_predicted_class_rule": 0, "identical_cards": 0}
        for i in range(data[c]["prototta"]["n"]):
            recs = {m: data[c][m]["samples"][i] for m in METHODS}
            why = eligible(recs)
            if why:
                skip[c][why] += 1
                continue
            cells[(c, stratum_of(recs))].append(i)
    pop = {s[0]: sum(len(cells[(c, s[0])]) for c in corruptions) for s in STRATA}
    n_pop = sum(pop.values())
    shares = {k: v / n_pop for k, v in pop.items()}
    quota = largest_remainder(shares, N_ITEMS)

    rng = np.random.default_rng(args.seed)
    extra = rng.choice(corruptions, size=N_ITEMS - len(corruptions), replace=False).tolist()
    corr_slots = corruptions + extra
    strat_slots = [k for k in quota for _ in range(quota[k])]
    for _ in range(1000):   # random table with fixed margins; every cell must be non-empty
        perm = rng.permutation(len(strat_slots))
        slots = [(corr_slots[k], strat_slots[perm[k]]) for k in range(N_ITEMS)]
        if all(cells[s] for s in slots):
            break
    else:
        raise SystemExit("no feasible table")

    used = set()
    if args.exclude_items and Path(args.exclude_items).exists():
        prev = json.loads(Path(args.exclude_items).read_text())
        used |= {it["path"] for it in prev["items"]}
        if prev.get("tutorial"):
            used.add(prev["tutorial"]["path"])
    orders = {k: rng.permutation(v).tolist() for k, v in cells.items()}
    cursor = {k: 0 for k in cells}
    class_names = data[corruptions[0]]["prototta"]["class_names"]

    def draw(cell):
        while cursor[cell] < len(orders[cell]):
            i = orders[cell][cursor[cell]]
            cursor[cell] += 1
            path = data[cell[0]]["prototta"]["samples"][i]["path"]
            if path in used:
                continue
            used.add(path)
            return i
        raise SystemExit(f"cell {cell} exhausted")

    def make_item(item_id, c, i):
        recs = {m: data[c][m]["samples"][i] for m in METHODS}
        label = recs["prototta"]["label"]
        return {"item_id": item_id, "corruption": c, "sample_index": i, "path": recs["prototta"]["path"],
                "dataset_root": data[c]["prototta"]["dataset_root"], "stratum": stratum_of(recs),
                "gt_label": label, "gt_class": class_names[label],          # analysis only, never shown
                "methods": {m: {"pred": r["pred"], "pred_class": class_names[r["pred"]], "correct": r["pred"] == label,
                                "conf": r["conf"], "top": r["top"]} for m, r in recs.items()}}

    slot_list = []
    for k, (c, s) in enumerate(slots):          # primaries first, so backups never pre-empt a primary bird
        slot_list.append({"slot": f"pw{k:02d}", "corruption": c, "stratum": s,
                          "candidates": [make_item(f"pw{k:02d}c0", c, draw((c, s)))]})
    for sl in slot_list:
        for b in range(1, N_BACKUP + 1):
            sl["candidates"].append(make_item(f"{sl['slot']}c{b}", sl["corruption"], draw((sl["corruption"], sl["stratum"]))))
    tut_cell = list(cells)[int(rng.integers(len(cells)))]
    while not cells[tut_cell]:
        tut_cell = list(cells)[int(rng.integers(len(cells)))]
    ti = draw(tut_cell)
    tm = str(rng.choice(METHODS))
    tr = data[tut_cell[0]][tm]["samples"][ti]
    tutorial = {"item_id": "tutorial", "corruption": tut_cell[0], "sample_index": ti, "path": tr["path"],
                "dataset_root": data[tut_cell[0]][tm]["dataset_root"], "method": tm, "top": tr["top"]}
    out = {"design": "pairwise", "methods_compared": list(METHODS), "seed": args.seed, "n_items": N_ITEMS,
           "rule": __doc__, "population_counts": pop, "population_shares": shares, "stratum_quota": quota,
           "extra_corruptions": extra, "skip_log": skip, "exclude_items": args.exclude_items,
           "export_sha256": provenance, "slots": slot_list, "tutorial": tutorial}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"population": pop, "quota": quota, "extra_corruptions": extra,
                      "slots": [(s["slot"], s["corruption"], s["stratum"]) for s in slot_list]}, indent=1))


if __name__ == "__main__":
    main()
