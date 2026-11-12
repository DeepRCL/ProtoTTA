#!/usr/bin/env python
"""Part-annotation interpretability metrics (Huang et al., ICCV 2023) for TTA on CUB-200-C,
computed for Unadapted / EATA / ProtoTTA (Table 4 rows 1-2, Appendix O).

Part set of a prototype on an image: the CUB part keypoints (visible, mapped to the
224x224 model input as in cub_parts.py) that lie in one of the prototype's active
sub-patch cells or in a neighbouring cell ("near").
Only ground-truth-class prototypes are used (10 per class), as in Huang et al.

* Stability (TTA version): for each corrupted test image i and each GT-class
  prototype p, does the part set under the method on the corrupted image equal the
  part set of the UNADAPTED model on the CLEAN image? Pairs whose clean part set is
  empty are excluded (reported). Higher = adaptation restores the clean part
  grounding. Paired method differences: cluster bootstrap over source images.
* Consistency: for each prototype p, the fraction of its class's test images on
  which each part appears in its part set; p is consistent if the most frequent
  part reaches mu = 0.8. Consistency = share of consistent prototypes. Reported on
  clean images (reference) and per method on corrupted images. Paired differences:
  bootstrap over classes (each class resampled with all 10 of its prototypes, since
  they share the class's images); inference is across classes.
Inputs must pass the same verification/hash checks as the other analyses.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PV = HERE.parent
sys.path.insert(0, str(HERE))
from cub_parts import GRID, PATCH, SLOT_ACTIVE, load_keypoints  # noqa: E402

METHODS = ("unadapted", "eata", "prototta")
SEVERITY = 5
MU = 0.8


def part_set(proto, pts, mode):
    parts = set()
    for flat, s in zip(proto["loc"], proto["slots"]):
        if s <= SLOT_ACTIVE:
            continue
        r, c = divmod(int(flat), GRID)
        x0, y0 = c * PATCH, r * PATCH
        cx, cy = x0 + PATCH / 2, y0 + PATCH / 2
        for pid, px, py in pts:
            inside = x0 <= px < x0 + PATCH and y0 <= py < y0 + PATCH
            near = max(abs(px - cx), abs(py - cy)) <= PATCH * 1.5
            if (inside if mode == "strict" else near):
                parts.add(pid)
    return frozenset(parts)


def boot_ci(diff, clusters=None, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if clusters is None:
        stats = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    else:
        uniq, inv = np.unique(clusters, return_inverse=True)
        sums, counts = np.bincount(inv, weights=diff), np.bincount(inv)
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(uniq), len(uniq))
            stats.append(sums[idx].sum() / counts[idx].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def consistency(per_proto_sets):
    """per_proto_sets: {p: [part_set per class image]} -> {p: 0/1 consistent}"""
    out = {}
    for p, sets in per_proto_sets.items():
        if not sets:
            continue
        counts = {}
        for s in sets:
            for pid in s:
                counts[pid] = counts.get(pid, 0) + 1
        out[p] = int(bool(counts) and max(counts.values()) / len(sets) >= MU)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default=str(PV / "results/explanation_eval/states_gt"),
                    help="exports written by export_states.py --with_gt_protos")
    ap.add_argument("--clean", default=str(PV / "results/explanation_eval/clean_reference.json"),
                    help="written by export_clean_reference.py")
    ap.add_argument("--out", default=str(PV / "results/explanation_eval/consistency_stability.json"))
    args = ap.parse_args()
    root = Path(args.states_root)
    ver = {}
    for m in METHODS:
        v = json.loads((root / m / "_verification.json").read_text())
        if not v.get("passed") or v.get("severity") != SEVERITY or v.get("gate") != "exact_count_v2":
            raise SystemExit(f"{m} export did not pass verification")
        ver[m] = v
    kp = load_keypoints()
    clean_path = Path(args.clean)
    cver = json.loads(clean_path.with_suffix(".verification.json").read_text())
    if not cver.get("passed") or hashlib.sha256(clean_path.read_bytes()).hexdigest() != cver["file_sha256"]:
        raise SystemExit("clean reference failed verification or changed after it")
    clean = json.loads(clean_path.read_text())["samples"]
    clean_by_path = {r["path"]: r for r in clean}
    assert len(clean_by_path) == len(clean)
    proto_cls = {}
    for r in clean:
        for g in r["gt_protos"]:
            assert proto_cls.setdefault(g["p"], r["label"]) == r["label"]

    corruptions = sorted(p.stem[:-3] for p in (root / "prototta").glob(f"*_s{SEVERITY}.json"))
    assert len(corruptions) == 13, corruptions
    report = {"definition": __doc__, "mu": MU, "modes": {}}
    for mode in ("near",):
        # clean reference part sets and clean consistency
        clean_sets, clean_pp = {}, {}
        for r in clean:
            pts = kp.get(r["path"], [])
            for g in r["gt_protos"]:
                s = part_set(g, pts, mode)
                clean_sets[(r["path"], g["p"])] = s
                clean_pp.setdefault(g["p"], []).append(s)
        clean_cons = consistency(clean_pp)
        stab = {m: [] for m in METHODS}
        clusters, n_excluded_empty = [], 0
        cons = {m: {} for m in METHODS}
        per_corr = {}
        for c in corruptions:
            data = {}
            for m in METHODS:
                f = root / m / f"{c}_s{SEVERITY}.json"
                if hashlib.sha256(f.read_bytes()).hexdigest() != ver[m]["export_sha256"].get(f.name):
                    raise SystemExit(f"{m}/{f.name} does not match its verified hash")
                data[m] = json.loads(f.read_text())["samples"]
            paths = [s["path"] for s in data["prototta"]]
            assert all([s["path"] for s in data[m]] == paths for m in METHODS), c
            assert set(paths) == set(clean_by_path), f"{c}: corrupted and clean image sets differ"
            pp = {m: {} for m in METHODS}
            corr_stab = {m: [] for m in METHODS}
            for i, path in enumerate(paths):
                pts = kp.get(path, [])
                cref = clean_by_path[path]
                assert cref["label"] == data["prototta"][i]["label"]
                recs = {m: {g["p"]: g for g in data[m][i]["gt_protos"]} for m in METHODS}
                assert all(set(recs[m]) == {g["p"] for g in cref["gt_protos"]} for m in METHODS), (c, path)
                for g in cref["gt_protos"]:
                    ref = clean_sets[(path, g["p"])]
                    sets = {m: part_set(recs[m][g["p"]], pts, mode) for m in METHODS}
                    for m in METHODS:
                        pp[m].setdefault(g["p"], []).append(sets[m])
                    if not ref:
                        n_excluded_empty += 1
                        continue
                    for m in METHODS:
                        v = float(sets[m] == ref)
                        stab[m].append(v)
                        corr_stab[m].append(v)
                    clusters.append(path)
            for m in METHODS:
                cons_c = consistency(pp[m])
                for p, v in cons_c.items():
                    cons[m].setdefault(p, []).append(v)
            per_corr[c] = {m: {"stability": float(np.mean(corr_stab[m])),
                               "consistency": float(np.mean(list(consistency(pp[m]).values())))}
                           for m in METHODS}
        clusters = np.array(clusters)
        cons_mean = {m: {p: float(np.mean(v)) for p, v in cons[m].items()} for m in METHODS}
        protos = sorted(cons_mean["prototta"])
        res = {
            "n_stability_pairs": int(len(clusters)), "n_excluded_empty_clean_partset": n_excluded_empty,
            "clean_consistency": float(np.mean(list(clean_cons.values()))),
            "stability": {m: float(np.mean(stab[m])) for m in METHODS},
            "consistency_corrupted": {m: float(np.mean([cons_mean[m][p] for p in protos])) for m in METHODS},
            "paired": {}, "per_corruption": per_corr,
        }
        for a, b in (("prototta", "eata"), ("prototta", "unadapted"), ("eata", "unadapted")):
            res["paired"][f"{a}-{b}"] = {
                "stability": boot_ci(np.array(stab[a]) - np.array(stab[b]), clusters),
                # classes are resampled with all their prototypes (they share the class's images)
                "consistency": boot_ci([cons_mean[a][p] - cons_mean[b][p] for p in protos],
                                       np.array([proto_cls[p] for p in protos])),
            }
        report["modes"][mode] = res
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_corruption"}
                      for k, v in report["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
