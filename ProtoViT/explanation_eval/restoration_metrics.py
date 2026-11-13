#!/usr/bin/env python
"""Closeness of each method's explanation to the clean explanation: the eight pre-specified measures of
Appendix O (Table 4 rows 3-5).

Reference = the unadapted source model on the CLEAN image (export_clean_reference.py --full).
For every corrupted test pair (5,794 images x 13 corruptions) and method m in
{unadapted, eata, prototta} we measure how close m's explanation on the corrupted image
is to the clean explanation:
  P1 card_agreement    |top5_m & top5_clean| / 5 (predicted-class card prototypes)
  P2 card_part_jaccard Jaccard of the CUB parts covered ("near") by the card's active
                       sub-patches, method-on-corrupted vs clean (clean set empty -> excluded)
  P3 stability         Huang et al. "near" (recomputed per (image, GT-class prototype) pair with
                       the same part_set code; must equal consistency_stability.json)
  S1 any10_agreement   |top10any_m & top10any_clean| / 10
  S2 gt_spearman       Spearman rho of the 10 GT-class prototype activations vs clean
  S3 gt_act_mae        mean |act_m - act_clean| over the 10 GT-class prototypes (lower better)
  S4a gt_loc_exact     share of active GT-class sub-patches at exactly the clean cell
  S4b gt_loc_within1   ... within one cell (Chebyshev)
Every measure is also reported on the images where all three methods keep the clean prediction.
ProtoTTA - EATA: cluster bootstrap over source images (2,000, seed 0).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PV = HERE.parent
sys.path.insert(0, str(HERE))
from consistency_stability import part_set  # noqa: E402
from cub_parts import GRID, SLOT_ACTIVE, load_keypoints  # noqa: E402

METHODS = ("unadapted", "eata", "prototta")
PAIRS = (("prototta", "eata"),)
SEVERITY = 5
GATE = "exact_count_v2"
METRICS = ("card_agreement", "card_part_jaccard", "any10_agreement", "gt_spearman", "gt_act_mae",
           "gt_loc_exact", "gt_loc_within1")          # one row per (image, corruption)
ALL_METRICS = METRICS + ("stability",)                # stability: one row per (image, corruption, GT prototype)
LOWER_IS_BETTER = {"gt_act_mae"}
N_BOOT = 2000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def card_parts(rec, pts):
    s = set()
    for t in rec["top"]:
        s |= part_set(t, pts, "near")
    return s


def gt_metrics(gm, gc):
    """gm, gc: {p: record} for the 10 GT-class prototypes (method on corrupted, clean)."""
    ps = sorted(gc)
    a_m = np.array([gm[p]["act"] for p in ps])
    a_c = np.array([gc[p]["act"] for p in ps])
    rho = spearmanr(a_m, a_c).statistic
    exact, within1, n = 0, 0, 0
    for p in ps:
        sm, sc = gm[p]["slots"], gc[p]["slots"]
        assert np.allclose(sm, sc, atol=1e-6), p   # slot selection is not adapted by any method
        for lm, lc, s in zip(gm[p]["loc"], gc[p]["loc"], sc):
            if s <= SLOT_ACTIVE:
                continue
            (rm, cm), (rc, cc) = divmod(int(lm), GRID), divmod(int(lc), GRID)
            exact += int(lm == lc)
            within1 += int(max(abs(rm - rc), abs(cm - cc)) <= 1)
            n += 1
    return {"gt_spearman": float(rho) if np.isfinite(rho) else np.nan,
            "gt_act_mae": float(np.abs(a_m - a_c).mean()),
            "gt_loc_exact": exact / n if n else np.nan,
            "gt_loc_within1": within1 / n if n else np.nan}


def boot(diff, clusters, seed=0):
    """Cluster bootstrap of the mean paired difference (NaNs dropped pairwise beforehand)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=diff), np.bincount(inv)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(uniq), len(uniq))
        stats[b] = sums[idx].sum() / counts[idx].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(diff)), "n_clusters": int(len(uniq))}


def summarize(tab, subset=None):
    """tab: {'vals': {m: array}, 'cl', 'corr', 'same', 'valid'}. One shared population for all
    methods and comparisons: rows valid (finite) for every method, intersected with the subset."""
    mask = tab["valid"] if subset is None else tab["valid"] & tab[subset]
    if not mask.any():
        return {"n": 0, "means": None, "paired": None}
    return {"n": int(mask.sum()), "n_clusters": int(len(np.unique(tab["cl"][mask]))),
            "means": {m: float(tab["vals"][m][mask].mean()) for m in METHODS},
            "paired": {f"{a}-{b}": boot(tab["vals"][a][mask] - tab["vals"][b][mask], tab["cl"][mask])
                       for a, b in PAIRS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default=str(PV / "results/explanation_eval/states_gt"),
                    help="exports written by export_states.py --with_gt_protos")
    ap.add_argument("--card_states_root", default=str(PV / "results/explanation_eval/states"),
                    help="exports the study cards came from; cards must be identical")
    ap.add_argument("--clean", default=str(PV / "results/explanation_eval/clean_reference_full.json"))
    ap.add_argument("--stability", default=str(PV / "results/explanation_eval/consistency_stability.json"))
    ap.add_argument("--out", default=str(PV / "results/explanation_eval/restoration_metrics.json"))
    ap.add_argument("--max_corruptions", type=int, default=None, help="smoke test only")
    args = ap.parse_args()

    roots = {"gt": Path(args.states_root), "card": Path(args.card_states_root)}
    ver = {}
    for kind, root in roots.items():
        for m in METHODS:
            v = json.loads((root / m / "_verification.json").read_text())
            if not v.get("passed") or v.get("severity") != SEVERITY or v.get("gate") != GATE:
                raise SystemExit(f"{root}/{m} export did not pass verification")
            ver[(kind, m)] = v
    clean_path = Path(args.clean)
    cver = json.loads(clean_path.with_suffix(".verification.json").read_text())
    if not cver.get("passed") or sha256(clean_path) != cver["file_sha256"]:
        raise SystemExit("clean full reference failed verification or changed after it")
    clean = {r["path"]: r for r in json.loads(clean_path.read_text())["samples"]}
    kp = load_keypoints()

    corruptions = sorted(p.stem[:-3] for p in (roots["gt"] / "prototta").glob(f"*_s{SEVERITY}.json"))
    assert len(corruptions) == 13, corruptions
    if args.max_corruptions:
        corruptions = corruptions[:args.max_corruptions]

    vals = {m: {k: [] for k in METRICS} for m in METHODS}
    clusters, corr_of, same_pred = [], [], []
    n_no_kp = n_clean_parts_empty = n_card_checked = 0
    stab = {m: [] for m in METHODS}
    stab_clusters, stab_corr, stab_same = [], [], []
    for c in corruptions:
        data = {}
        for kind, root in roots.items():
            for m in METHODS:
                f = root / m / f"{c}_s{SEVERITY}.json"
                if sha256(f) != ver[(kind, m)]["export_sha256"].get(f.name):
                    raise SystemExit(f"{f} does not match its verified hash")
                data[(kind, m)] = json.loads(f.read_text())["samples"]
        paths = [s["path"] for s in data[("gt", "prototta")]]
        assert set(paths) == set(clean), c
        for key, recs in data.items():
            assert [s["path"] for s in recs] == paths, (key, c)
        for i, path in enumerate(paths):
            cref = clean[path]
            recs = {m: data[("gt", m)][i] for m in METHODS}
            for m in METHODS:
                r, rc = recs[m], data[("card", m)][i]
                assert r["state_check"] and r["label"] == cref["label"], (c, m, path)
                # the gt export must reproduce the study-card export exactly
                assert r["pred"] == rc["pred"] and \
                    [(t["p"], t["loc"]) for t in r["top"]] == [(t["p"], t["loc"]) for t in rc["top"]], (c, m, path)
                n_card_checked += 1
            pts = kp.get(path) or []
            if not pts:
                n_no_kp += 1
            clean_card_parts = card_parts(cref, pts)
            gc = {g["p"]: g for g in cref["gt_protos"]}
            ctop = {t["p"] for t in cref["top"]}
            cany = {t["p"] for t in cref["top_any"]}
            for m in METHODS:
                r = recs[m]
                vals[m]["card_agreement"].append(len({t["p"] for t in r["top"]} & ctop) / 5)
                vals[m]["any10_agreement"].append(len({t["p"] for t in r["top_any"]} & cany) / 10)
                if clean_card_parts:
                    mp = card_parts(r, pts)
                    vals[m]["card_part_jaccard"].append(len(mp & clean_card_parts) / len(mp | clean_card_parts))
                else:
                    vals[m]["card_part_jaccard"].append(np.nan)
                gm = {g["p"]: g for g in r["gt_protos"]}
                assert set(gm) == set(gc), (c, m, path)
                for k, v in gt_metrics(gm, gc).items():
                    vals[m][k].append(v)
            same = all(recs[m]["pred"] == cref["pred"] for m in METHODS)
            for g in cref["gt_protos"]:   # Huang et al. stability, pair level, same order as consistency_stability.py
                ref = part_set(g, pts, "near")
                if not ref:
                    continue
                for m in METHODS:
                    gm = next(x for x in recs[m]["gt_protos"] if x["p"] == g["p"])
                    stab[m].append(float(part_set(gm, pts, "near") == ref))
                stab_clusters.append(path)
                stab_corr.append(c)
                stab_same.append(same)
            n_clean_parts_empty += int(not clean_card_parts)
            clusters.append(path)
            corr_of.append(c)
            same_pred.append(same)
        print(f"[restoration] {c} done ({len(clusters)} pairs)", flush=True)

    clusters, corr_of, same_pred = np.array(clusters), np.array(corr_of), np.array(same_pred, dtype=bool)
    tables = {}
    for k in METRICS:
        v = {m: np.array(vals[m][k], dtype=float) for m in METHODS}
        tables[k] = {"vals": v, "cl": clusters, "corr": corr_of, "same": same_pred,
                     "valid": np.logical_and.reduce([np.isfinite(v[m]) for m in METHODS])}
    sv = {m: np.array(stab[m], dtype=float) for m in METHODS}
    tables["stability"] = {"vals": sv, "cl": np.array(stab_clusters), "corr": np.array(stab_corr),
                           "same": np.array(stab_same, dtype=bool), "valid": np.ones(len(stab_clusters), dtype=bool)}
    report = {
        "definition": __doc__,
        "n_pairs": int(len(clusters)), "n_pairs_no_visible_keypoint": n_no_kp,
        "n_card_records_crosschecked_vs_study_exports": n_card_checked,
        "n_rows": {k: int(len(t["valid"])) for k, t in tables.items()},
        "n_excluded_undefined": {k: int((~t["valid"]).sum()) for k, t in tables.items()},
        "n_excluded_card_part_jaccard_clean_card_covers_no_part": n_clean_parts_empty,
        "clean_accuracy": cver["clean_accuracy"],
        "all": {k: summarize(t) for k, t in tables.items()},
        "same_prediction_as_clean_all_methods": {k: summarize(t, "same") for k, t in tables.items()},
        "per_corruption": {}, "n_corruptions_prototta_better_than_eata": {},
    }
    for k, t in tables.items():
        per, wins = {}, 0
        for c in corruptions:
            msk = t["valid"] & (t["corr"] == c)
            per[c] = {m: (float(t["vals"][m][msk].mean()) if msk.any() else None) for m in METHODS}
            if msk.any():
                d = per[c]["prototta"] - per[c]["eata"]
                wins += int(d < 0 if k in LOWER_IS_BETTER else d > 0)
        report["per_corruption"][k] = per
        report["n_corruptions_prototta_better_than_eata"][k] = wins
    report["n_stability_pairs"] = int(len(stab_clusters))
    if not args.max_corruptions:   # must reproduce the separately computed stability
        st = json.loads(Path(args.stability).read_text())["modes"]["near"]
        assert st["n_stability_pairs"] == len(stab_clusters), (st["n_stability_pairs"], len(stab_clusters))
        if len(stab_clusters):
            for m in METHODS:
                assert abs(st["stability"][m] - report["all"]["stability"]["means"][m]) < 1e-12, m
        report["stability_matches_consistency_stability_json"] = True
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print("[restoration] written", args.out, flush=True)
    for scope in ("all", "same_prediction_as_clean_all_methods"):
        print(f"\n## {scope}")
        for k in ALL_METRICS:
            r = report[scope][k]
            if not r["n"]:
                continue
            m, d = r["means"], r["paired"]["prototta-eata"]
            wins = (f"; ProtoTTA better than EATA on {report['n_corruptions_prototta_better_than_eata'][k]}/"
                    f"{len(corruptions)} corruptions" if scope == "all" else "")
            print(f"{k}: unadapted {m['unadapted']:.4f}  eata {m['eata']:.4f}  prototta {m['prototta']:.4f}  "
                  f"ProtoTTA-EATA {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] (n={r['n']}){wins}")


if __name__ == "__main__":
    main()
