#!/usr/bin/env python
"""Label-free, seeded draw of items (test photos) from the exported states.

Rule (uses neither correctness nor which method's explanation looks better):
  1. For every corruption (sorted by name) draw a seeded random permutation of
     the test-image indices.
  2. An index is eligible iff (a) the per-sample state check passed for all three
     methods (captured pre-update state reproduces the returned logits), (b) each
     method has 5 prototypes chosen by the predicted-class rule (no fallback), (c)
     the three explanations are not all identical as displayed (same prototypes
     with the same active sub-patch locations), and (d) its source bird image has
     not been used by another item (or by --exclude_items).
  Exclusion counts per reason are reported in skip_log.
  3. --mode vlm_large: the first --per_corruption eligible indices of every corruption
     (pool of the 104-image VLM comparison, make_items_vlm104.py).
     --mode human: the first eligible index of each corruption (13 items), then N_EXTRA
     more items from N_EXTRA corruptions drawn at random, N_REPEAT repeated items and a
     tutorial item (next eligible index of a random corruption, random method). Its photos
     are excluded from the later draws (select_pairwise.py and --mode vlm_large).
Ground truth is stored for the offline analysis only and is never shown.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PV = HERE.parent
SEED = 20261106
N_EXTRA = 2
N_REPEAT = 2
SEVERITY = 5
SLOT_ACTIVE = 0.5  # same visualization threshold as render_cards.py
METHODS = ("unadapted", "eata", "prototta")


def active_locs(proto):
    # (slot index, location) for every active sub-prototype: exactly what both card halves draw
    return tuple((k, int(f)) for k, (f, s) in enumerate(zip(proto["loc"], proto["slots"])) if s > SLOT_ACTIVE)


def card_signature(rec):
    # exactly the evidence drawn on a card: prototype identity + active sub-patch boxes
    return tuple((t["p"], active_locs(t)) for t in rec["top"])


def source_image(path):
    # "<class_dir>/<file>": the same bird image exists under every corruption
    return path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default=str(PV / "results/explanation_eval/states"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["human", "vlm_large"], default="human",
                    help="human: 13+N_EXTRA items, repeats, tutorial; "
                         "vlm_large: --per_corruption items per corruption, no repeats/tutorial")
    ap.add_argument("--per_corruption", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude_items", default=None,
                    help="items.json whose source images (incl. tutorial) must not be reused")
    args = ap.parse_args()
    root = Path(args.states_root)

    suffix = f"_s{SEVERITY}"
    corruptions = sorted(p.stem[: -len(suffix)] for p in (root / "prototta").glob(f"*{suffix}.json"))
    if len(corruptions) != 13:
        raise SystemExit(f"Expected 13 corruptions, found {len(corruptions)}: {corruptions}")

    verification = {}
    for m in METHODS:
        ver = json.loads((root / m / "_verification.json").read_text())
        if not ver.get("passed") or ver.get("severity") != SEVERITY or ver.get("gate") != "exact_count_v2":
            raise SystemExit(f"Export for {m} did not pass verification: {ver.get('failures')}")
        verification[m] = ver
        print(f"[verify] {m}: passed; max |acc - reference| = {ver['max_abs_acc_delta']:.6f}")

    data, provenance = {}, {}
    for c in corruptions:
        data[c] = {}
        for m in METHODS:
            f = root / m / f"{c}{suffix}.json"
            digest = file_sha256(f)
            if verification[m]["export_sha256"].get(f.name) != digest:
                raise SystemExit(f"{m}/{f.name} changed after verification")
            provenance[f"{m}/{f.name}"] = digest
            data[c][m] = json.loads(f.read_text())
            assert data[c][m]["severity"] == SEVERITY, (c, m)
        n = data[c]["prototta"]["n"]
        for m in METHODS:
            d = data[c][m]
            assert d["n"] == n == len(d["samples"]), (c, m)
            assert [s["path"] for s in d["samples"]] == [s["path"] for s in data[c]["prototta"]["samples"]], (c, m)
            assert [s["label"] for s in d["samples"]] == [s["label"] for s in data[c]["prototta"]["samples"]], (c, m)

    rng = np.random.default_rng(args.seed)
    perms = {c: rng.permutation(data[c]["prototta"]["n"]).tolist() for c in corruptions}
    cursor = {c: 0 for c in corruptions}
    used_sources = set()
    if args.exclude_items:
        prev = json.loads(Path(args.exclude_items).read_text())
        used_sources |= {source_image(it["path"]) for it in prev["items"]}
        if prev.get("tutorial"):
            used_sources.add(source_image(prev["tutorial"]["path"]))
    skip_log = {c: {"state_check": 0, "not_predicted_class_rule": 0, "all_identical": 0, "source_reused": 0}
                for c in corruptions}

    def next_eligible(c):
        while cursor[c] < len(perms[c]):
            i = perms[c][cursor[c]]
            cursor[c] += 1
            recs = {m: data[c][m]["samples"][i] for m in METHODS}
            if not all(r["state_check"] for r in recs.values()):
                skip_log[c]["state_check"] += 1
                continue
            if not all(r["top_rule"] == "predicted_class" and len(r["top"]) == 5
                       and all(t["cls"] == r["pred"] and t["w"] > 0 for t in r["top"])
                       for r in recs.values()):
                skip_log[c]["not_predicted_class_rule"] += 1
                continue
            if len({card_signature(r) for r in recs.values()}) == 1:
                skip_log[c]["all_identical"] += 1
                continue
            src = source_image(recs["prototta"]["path"])
            if src in used_sources:
                skip_log[c]["source_reused"] += 1
                continue
            used_sources.add(src)
            return i, recs
        raise RuntimeError(f"No eligible sample left for {c}")

    if args.mode == "human":
        picks = [(c, *next_eligible(c)) for c in corruptions]
        extra_corruptions = rng.choice(corruptions, size=N_EXTRA, replace=False).tolist()
        picks += [(c, *next_eligible(c)) for c in extra_corruptions]
        id_fmt = "it{:02d}"
    else:
        picks = [(c, *next_eligible(c)) for c in corruptions for _ in range(args.per_corruption)]
        extra_corruptions = []
        id_fmt = "vl{:03d}"

    class_names = data[corruptions[0]]["prototta"]["class_names"]
    items = []
    for k, (c, i, recs) in enumerate(picks):
        label = recs["prototta"]["label"]
        items.append({
            "item_id": id_fmt.format(k),
            "corruption": c,
            "sample_index": i,
            "path": recs["prototta"]["path"],
            "dataset_root": data[c]["prototta"]["dataset_root"],
            "gt_label": label,                      # analysis only, never shown
            "gt_class": class_names[label],         # analysis only, never shown
            "methods": {m: {"pred": r["pred"], "pred_class": class_names[r["pred"]],
                            "correct": r["pred"] == label, "conf": r["conf"],
                            "top": r["top"]} for m, r in recs.items()},
        })
    repeat_ids, tutorial = [], None
    if args.mode == "human":
        repeat_ids = rng.choice([it["item_id"] for it in items], size=N_REPEAT, replace=False).tolist()
        tut_corruption = str(rng.choice(corruptions))
        tut_i, tut_recs = next_eligible(tut_corruption)
        tut_method = str(rng.choice(METHODS))
        tutorial = {
            "item_id": "tutorial", "corruption": tut_corruption, "sample_index": tut_i,
            "path": tut_recs[tut_method]["path"], "dataset_root": data[tut_corruption][tut_method]["dataset_root"],
            "method": tut_method, "top": tut_recs[tut_method]["top"],
        }

    out = {
        "mode": args.mode, "seed": args.seed, "n_items": len(items), "extra_corruptions": extra_corruptions,
        "exclude_items": args.exclude_items,
        "repeat_item_ids": repeat_ids, "tutorial": tutorial, "rule": __doc__, "skip_log": skip_log,
        "export_sha256": provenance, "items": items,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(items)} items (+{len(repeat_ids)} repeats) to {args.out}")
    print("skips:", json.dumps(skip_log))


if __name__ == "__main__":
    main()
