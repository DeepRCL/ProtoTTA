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
  identical (X, X), X a seeded coin per pair  -> the true answer is "no change".

Ground truth per state (what the board shows, plus offline label-based criteria):
  boxes          (prototype rank, 14x14 cell) for every active sub-patch of the top-5
                 predicted-class prototypes (slot > 0.5; exactly the boxes drawn);
  box region     BACKGROUND if < 50% of the cell is bird in the CUB segmentation mask
                 (pixel = bird if >= 3 of 5 annotators, value >= 128, after the same
                 bbox crop + resize to 224 as the model input); otherwise the coarse
                 group of the nearest visible CUB keypoint (Euclidean, cell centre):
                 HEAD {beak, crown, forehead, eyes, nape, throat}, BODY {back, breast,
                 belly}, WING, TAIL, LEGS;
  dominant       region(s) with the most boxes (ties -> all tied regions), share;
  cells          set of boxed cells (focus-shift Jaccard);
  pictures       set of sha256 of the training-example pictures shown (290 groups of
                 prototypes share a picture, so identity is judged on pictures);
  n_competing    entries of the top-10 any-class list whose class != the state's prediction;
  offline only   correct, PCA-W (paper Eq. 8, k = 10, true class), part-hit (near),
                 background share.
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
from paths import CHECKPOINT, CUB_SEGMENTATIONS  # noqa: E402

SEG = CUB_SEGMENTATIONS
METHODS = ("unadapted", "prototta")
STRATA = ("broken", "fixed", "both_wrong", "both_right")   # rarest first (source-photo uniqueness)
REGIONS = ("HEAD", "BODY", "WING", "TAIL", "LEGS", "BACKGROUND")
GROUP_OF = {2: "HEAD", 5: "HEAD", 6: "HEAD", 7: "HEAD", 10: "HEAD", 11: "HEAD", 15: "HEAD",
            1: "BODY", 3: "BODY", 4: "BODY", 9: "WING", 13: "WING", 14: "TAIL", 8: "LEGS", 12: "LEGS"}
MASK_BIRD = 128        # >= 3 of 5 annotators
CELL_BIRD = 0.5        # fraction of a 16x16 cell that must be bird
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


def bird_cells(rel, bbox_of, img_id_of):
    """14x14 fraction of bird pixels per cell, in the model's 224 input frame."""
    with Image.open(lm.CUB / "images" / rel) as im:
        W0, H0 = im.size
    x, y, w, h = bbox_of[img_id_of[rel]]
    left, top = max(0, int(x)), max(0, int(y))
    right, bottom = min(W0, int(x) + int(w)), min(H0, int(y) + int(h))
    with Image.open(lm.CROPPED / rel) as im:   # same crop as the test images (asserted in lm too)
        assert im.size == (right - left, bottom - top), rel
    with Image.open(SEG / (rel[:-4] + ".png")) as m:
        m = m.convert("L")
        assert m.size == (W0, H0), (rel, m.size, (W0, H0))
        m = m.crop((left, top, right, bottom)).resize((lm.IMG, lm.IMG), Image.BILINEAR)
    bird = np.asarray(m, dtype=np.float32) >= MASK_BIRD
    return bird.reshape(lm.GRID, lm.PATCH, lm.GRID, lm.PATCH).mean(axis=(1, 3))


def box_region(flat, cov, pts):
    r, c = divmod(int(flat), lm.GRID)
    if cov[r, c] < CELL_BIRD:
        return "BACKGROUND"
    cx, cy = c * lm.PATCH + lm.PATCH / 2, r * lm.PATCH + lm.PATCH / 2
    pid = min(pts, key=lambda t: ((t[1] - cx) ** 2 + (t[2] - cy) ** 2, t[0]))[0]
    return GROUP_OF[pid]


def state_gt(rec, cov, pts, W, pic_hash):
    boxes = [(k, int(f)) for k, p in enumerate(rec["top"]) for f, s in zip(p["loc"], p["slots"])
             if float(s) > lm.SLOT_ACTIVE]
    if not boxes:
        return None
    regions = [box_region(f, cov, pts) for _, f in boxes]
    counts = {g: regions.count(g) for g in REGIONS}
    mx = max(counts.values())
    hit = lm.sample_metrics(rec, pts)
    return {
        "pred": int(rec["pred"]), "correct": bool(rec["pred"] == rec["label"]),
        "n_boxes": len(boxes), "box_regions": regions, "region_counts": counts,
        "dominant": [g for g in REGIONS if counts[g] == mx], "dominant_share": mx / len(boxes),
        "cells": sorted({f for _, f in boxes}),
        "proto_ids": [int(p["p"]) for p in rec["top"]],
        "pictures": sorted({pic_hash[int(p["p"])] for p in rec["top"]}),
        "n_competing": int(sum(int(t["cls"]) != int(rec["pred"]) for t in rec["top_any"])),
        "background_share": counts["BACKGROUND"] / len(boxes),
        "part_hit_near": None if hit is None else hit[0],
        "pcaw": lm.pcaw(rec, W),
    }


def main():
    import torch
    sys.path.insert(0, str(PV))
    import model  # noqa: F401  (for torch.load)
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
    W = torch.load(CHECKPOINT, weights_only=False, map_location="cpu").last_layer.weight.detach().cpu().numpy()
    kp = lm.load_keypoints()
    images = dict(l.split() for l in (lm.CUB / "images.txt").read_text().splitlines())
    img_id_of = {rel: i for i, rel in images.items()}
    bbox_of = {k: tuple(map(float, v)) for k, *v in
               (l.split() for l in (lm.CUB / "bounding_boxes.txt").read_text().splitlines())}
    pic_hash = {p: sha256_bytes((PROTO_DIR / f"prototype-img-original{p}.png").read_bytes()) for p in range(W.shape[1])}
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
        pts = kp[rel]
        cov = bird_cells(rel, bbox_of, img_id_of)
        gt = {"unadapted": state_gt(u, cov, pts, W, pic_hash), "prototta": state_gt(p, cov, pts, W, pic_hash)}
        if any(v is None for v in gt.values()):
            n_rejected["no_box"] += 1
            return None
        states = {m: {"pred": int(r["pred"]), "top": r["top"], "top_any": r["top_any"]}
                  for m, r in (("unadapted", u), ("prototta", p))}
        root = PV / roots[c]
        with Image.open(root / rel) as im_c, Image.open(lm.CROPPED / rel) as im_k:
            assert im_c.size == im_k.size, (c, rel, im_c.size, im_k.size)   # same frame as masks/keypoints
        try:
            for m in METHODS:
                state_board(root, rel, states[m], class_names, bb, "STATE 1")
        except Exception as exc:  # noqa: BLE001  (recorded as a rejection count)
            n_rejected["render_error"] += 1
            print(f"[items] render error {c} {rel}: {type(exc).__name__}: {exc}", flush=True)
            return None
        used.add(rel)
        return {"corruption": c, "sample_index": i, "path": rel, "dataset_root": roots[c],
                "label": int(u["label"]), "states": states, "gt": gt,
                "bird_cell_fraction": np.round(cov, 4).tolist()}

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

    common = {"seed": SEED, "severity": SEVERITY, "methods": list(METHODS), "regions": list(REGIONS),
              "class_names": class_names, "export_sha256": export_sha,
              "segmentation_archive_sha256": "dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f",
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
