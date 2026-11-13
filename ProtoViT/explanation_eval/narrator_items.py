#!/usr/bin/env python
"""Narrator validation: select (unadapted, ProtoTTA) state pairs and compute ground truth.

Population: CUB-200-C severity 5, all 13 corruptions, the exported states of the
unadapted model and the final adaptive ProtoTTA (export_states.py, verified exports).

Sampling (seeded, label use = stratification only; the narrator never sees labels):
  strata by correctness  fixed (U wrong, P right) | broken (U right, P wrong) |
                         both_right | both_wrong
  quota                  PER_CELL pairs per (corruption, stratum) -> 13 x 4 x 4 = 208 pairs;
                         a shortfall in a cell is filled from the same stratum in other
                         corruptions (recorded).
  eligibility            visible CUB keypoints; both states show >= 1 active box; both
                         boards render; every source photo is used at most once overall.
  smoke pairs            N_SMOKE extra pairs from photos disjoint from the validation set,
                         written to a separate file (pipeline checks only; never analysed).
Each pair yields three presentations (STATE 1 -> STATE 2):
  main      (unadapted, prototta)
  swapped   (prototta, unadapted)
  identical (X, X), X a seeded coin per pair.

Ground truth per state (used by narrator_analysis.py; the training-example pictures are compared
there from the stored board states):
  cells          set of boxed cells: every active sub-patch of the top-5 predicted-class
                 prototypes (slot > 0.5; exactly the boxes drawn);
  n_competing    entries of the top-10 any-class list whose class != the state's prediction.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PV = HERE.parent
sys.path.insert(0, str(HERE))
import cub_parts as lm  # noqa: E402

METHODS = ("unadapted", "prototta")
STRATA = ("broken", "fixed", "both_wrong", "both_right")   # rarest first (source-photo uniqueness)
REGIONS = ("HEAD", "BODY", "WING", "TAIL", "LEGS", "BACKGROUND")   # answer options of the narrator prompt
PER_CELL = 4
N_SMOKE = 4
SEED = 20261108
SEVERITY = lm.SEVERITY


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def stratum(u_correct, p_correct):
    return {(False, True): "fixed", (True, False): "broken",
            (True, True): "both_right", (False, False): "both_wrong"}[(u_correct, p_correct)]


def load_states(root):
    """{corruption: {method: [samples]}} from the verified exports (hash-checked)."""
    ver = {}
    for m in METHODS:
        v = json.loads((root / m / "_verification.json").read_text())
        if not v.get("passed") or v.get("severity") != SEVERITY or v.get("gate") != "exact_count_v2":
            raise SystemExit(f"{m} export did not pass verification")
        ver[m] = v
    corruptions = sorted(p.stem[: -len(f"_s{SEVERITY}")] for p in (root / "prototta").glob(f"*_s{SEVERITY}.json"))
    if len(corruptions) != 13:
        raise SystemExit(f"expected 13 corruptions, found {corruptions}")
    data, meta = {}, {}
    for c in corruptions:
        data[c] = {}
        for m in METHODS:
            f = root / m / f"{c}_s{SEVERITY}.json"
            raw = f.read_bytes()
            if sha256_bytes(raw) != ver[m]["export_sha256"].get(f.name):
                raise SystemExit(f"{m}/{f.name} does not match its verified hash")
            d = json.loads(raw)
            assert d["severity"] == SEVERITY and d["n"] == len(d["samples"]), (m, c)
            data[c][m] = d["samples"]
            meta[(c, m)] = {"dataset_root": d["dataset_root"], "class_names": d["class_names"]}
        assert [s["path"] for s in data[c]["unadapted"]] == [s["path"] for s in data[c]["prototta"]], c
        assert meta[(c, "unadapted")] == meta[(c, "prototta")], c
    names = {tuple(v["class_names"]) for v in meta.values()}
    assert len(names) == 1, "class names differ between exports"
    return data, {c: meta[(c, "prototta")]["dataset_root"] for c in corruptions}, list(names.pop()), \
        {m: ver[m]["export_sha256"] for m in METHODS}


def state_gt(rec):
    boxes = [(k, int(f)) for k, p in enumerate(rec["top"]) for f, s in zip(p["loc"], p["slots"])
             if float(s) > lm.SLOT_ACTIVE]
    if not boxes:
        return None
    return {
        "cells": sorted({f for _, f in boxes}),
        "n_competing": int(sum(int(t["cls"]) != int(rec["pred"]) for t in rec["top_any"])),
    }


def main():
    from narrator_board import state_board
    from render_cards import PROTO_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default=str(PV / "results/explanation_eval/states"))
    ap.add_argument("--out", default=str(PV / "results/explanation_eval/narrator/items.json"))
    ap.add_argument("--smoke_out", default=str(PV / "results/explanation_eval/narrator/smoke_items.json"))
    args = ap.parse_args()
    out, smoke_out = Path(args.out), Path(args.smoke_out)
    if out.exists() or smoke_out.exists():
        raise SystemExit(f"{out} or {smoke_out} exists; refusing to redraw the validation set")

    data, roots, class_names, export_sha = load_states(Path(args.states_root))
    kp = lm.load_keypoints()
    bb = np.load(PROTO_DIR / "bb4.npy")

    # population of eligible-by-label pairs (rendering/GT checked lazily on draw)
    cells, population = {}, {s: 0 for s in STRATA}
    for c, d in data.items():
        for i, (u, p) in enumerate(zip(d["unadapted"], d["prototta"])):
            assert u["path"] == p["path"] and u["label"] == p["label"] and u["state_check"] and p["state_check"]
            if not kp.get(u["path"]):
                continue
            s = stratum(u["pred"] == u["label"], p["pred"] == p["label"])
            population[s] += 1
            cells.setdefault((c, s), []).append(i)
    rng = random.Random(SEED)
    order = {k: rng.sample(v, len(v)) for k, v in sorted(cells.items())}   # seeded draw order per cell
    used, pairs, shortfall, n_rejected = set(), [], [], {"no_box": 0, "render_error": 0, "photo_used": 0}

    def try_pair(c, i):
        u, p = data[c]["unadapted"][i], data[c]["prototta"][i]
        rel = u["path"]
        if rel in used:
            n_rejected["photo_used"] += 1
            return None
        gt = {"unadapted": state_gt(u), "prototta": state_gt(p)}
        if any(v is None for v in gt.values()):
            n_rejected["no_box"] += 1
            return None
        states = {m: {"pred": int(r["pred"]), "top": r["top"], "top_any": r["top_any"]}
                  for m, r in (("unadapted", u), ("prototta", p))}
        root = PV / roots[c]
        with Image.open(root / rel) as im_c, Image.open(lm.CROPPED / rel) as im_k:
            assert im_c.size == im_k.size, (c, rel, im_c.size, im_k.size)   # same frame as the keypoints
        try:
            for m in METHODS:
                state_board(root, rel, states[m], class_names, bb, "STATE 1")
        except Exception as exc:  # noqa: BLE001  (recorded as a rejection count)
            n_rejected["render_error"] += 1
            print(f"[items] render error {c} {rel}: {type(exc).__name__}: {exc}", flush=True)
            return None
        used.add(rel)
        return {"corruption": c, "sample_index": i, "path": rel, "dataset_root": roots[c],
                "label": int(u["label"]), "states": states, "gt": gt}

    pos = {k: 0 for k in order}

    def draw(c, s):
        lst = order.get((c, s), [])
        while pos[(c, s)] < len(lst):
            i = lst[pos[(c, s)]]
            pos[(c, s)] += 1
            got = try_pair(c, i)
            if got is not None:
                return got
        return None

    corruptions = sorted(data)
    for s in STRATA:
        missing = 0
        for c in corruptions:
            for _ in range(PER_CELL):
                got = draw(c, s) if (c, s) in pos else None
                if got is None:
                    missing += 1
                else:
                    pairs.append({**got, "stratum": s, "filled_from_other_corruption": False})
        fill_order = rng.sample(corruptions, len(corruptions))
        while missing:
            progressed = False
            for c in fill_order:
                if not missing:
                    break
                got = draw(c, s) if (c, s) in pos else None
                if got is not None:
                    pairs.append({**got, "stratum": s, "filled_from_other_corruption": True})
                    missing -= 1
                    progressed = True
            if not progressed:
                shortfall.append({"stratum": s, "unfilled": missing})
                break
    smoke = []
    for k in range(N_SMOKE):   # disjoint photos, one per stratum, random corruption
        s = STRATA[k % len(STRATA)]
        for c in rng.sample(corruptions, len(corruptions)):
            got = draw(c, s) if (c, s) in pos else None
            if got is not None:
                smoke.append({**got, "stratum": s, "filled_from_other_corruption": False})
                break

    def finish(plist, tag):
        prng = random.Random(f"{SEED}:{tag}")
        for k, pr in enumerate(plist):
            pr["pair_id"] = f"{tag}{k:03d}"
            pr["identical_state"] = prng.choice(METHODS)
        plan = []
        for pr in plist:
            plan += [{"presentation_id": f"{pr['pair_id']}:main", "pair_id": pr["pair_id"], "condition": "main",
                      "order": ["unadapted", "prototta"]},
                     {"presentation_id": f"{pr['pair_id']}:swapped", "pair_id": pr["pair_id"],
                      "condition": "swapped", "order": ["prototta", "unadapted"]},
                     {"presentation_id": f"{pr['pair_id']}:identical", "pair_id": pr["pair_id"],
                      "condition": "identical", "order": [pr["identical_state"]] * 2}]
        prng.shuffle(plan)
        return plan

    common = {"seed": SEED, "severity": SEVERITY, "methods": list(METHODS),
              "class_names": class_names, "export_sha256": export_sha,
              "definition": __doc__,
              "code_sha256": {f: sha256_bytes((HERE / f).read_bytes())
                              for f in ("narrator_items.py", "narrator_board.py", "render_cards.py",
                                        "cub_parts.py")}}
    plan = finish(pairs, "nv")
    smoke_plan = finish(smoke, "smoke")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**common, "kind": "validation", "population_strata": population,
                               "per_cell_quota": PER_CELL, "shortfall": shortfall, "rejections": n_rejected,
                               "pairs": pairs, "plan": plan}))
    smoke_out.write_text(json.dumps({**common, "kind": "smoke", "pairs": smoke, "plan": smoke_plan}))
    by = {}
    for pr in pairs:
        by[pr["stratum"]] = by.get(pr["stratum"], 0) + 1
    print(f"[items] {len(pairs)} pairs {by}, {len(plan)} presentations; population {population}; "
          f"filled-from-other {sum(p['filled_from_other_corruption'] for p in pairs)}; shortfall {shortfall}; "
          f"rejections {n_rejected}; smoke {len(smoke)} pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
