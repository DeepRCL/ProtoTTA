#!/usr/bin/env python
"""Analysis of the narrator validation (Table 5, right; Appendix P).

Grades the narrator's fixed-vocabulary answers against what the two boards show
(model internals; CUB masks/keypoints) and checks two controls (identical boards,
swapped order). Table 5 (right) reports, on the main and swapped presentations:
retrieved training examples changed (C4, accuracy), competing evidence rose/fell (C5,
accuracy on pairs with a decisive change), and focus moved (C3, AUC of the narrator's
YES/NO against the measured spatial shift of the boxed cells).
CIs: cluster bootstrap over pairs (one source photo per pair; all presentations of a pair
resampled together), stratified by correctness stratum (26 pairs per stratum), 5,000
resamples, percentile intervals.
Provenance: every shard's manifest, plan membership, record metadata and (by
re-rendering) every stimulus hash are checked; the run must be complete.
"""
import argparse
import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from narrator_items import REGIONS  # noqa: E402
from render_cards import PROTO_DIR  # noqa: E402

N_BOOT = 5000
ALPHA = 0.05
N_CLAIMS = 6                                # Bonferroni family for "above chance"
ANNOTATION_UNRELIABLE = {"elastic_transform"}   # pixels displaced: clean masks/keypoints misaligned
RELIABLE_ACC, RELIABLE_AUC = 0.80, 0.75
MAX_FALSE_CHANGE, MIN_SWAP_CONSISTENCY = 0.10, 0.80
TERMINAL = ("ok", "parse_failed")           # "error" (infrastructure) is re-run on resume
PER_SHARD_KEYS = {"shard_index", "plan_sha256", "stimuli_sha256"}
ORD = {"verdict": {"IMPROVED": 1, "NO_MEANINGFUL_CHANGE": 0, "WORSENED": -1},
       "competing_evidence": {"DECREASED": -1, "ABOUT_THE_SAME": 0, "INCREASED": 1}}
MIRROR = {"verdict": {"IMPROVED": "WORSENED", "WORSENED": "IMPROVED", "NO_MEANINGFUL_CHANGE": "NO_MEANINGFUL_CHANGE"},
          "competing_evidence": {"DECREASED": "INCREASED", "INCREASED": "DECREASED", "ABOUT_THE_SAME": "ABOUT_THE_SAME"}}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_json(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode())


# ----------------------------------------------------------------------------- ground truth
_PIC = {}


def picture(p):
    if p not in _PIC:
        _PIC[p] = sha256_bytes((PROTO_DIR / f"prototype-img-original{int(p)}.png").read_bytes())
    return _PIC[p]


def examples_label(s1, s2):
    """SAME iff the five tiles show the same pictures with the same multiplicities (any order);
    DIFFERENT iff no picture is shared; PARTLY otherwise."""
    a = collections.Counter(picture(t["p"]) for t in s1["top"])
    b = collections.Counter(picture(t["p"]) for t in s2["top"])
    if a == b:
        return "SAME"
    return "DIFFERENT" if not set(a) & set(b) else "PARTLY"


def dilate(cells, grid=14):
    out = set()
    for f in cells:
        r, c = divmod(int(f), grid)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if 0 <= r + dr < grid and 0 <= c + dc < grid:
                    out.add((r + dr) * grid + c + dc)
    return out


def region_tv(g1, g2):
    a = np.array([g1["region_counts"][r] for r in REGIONS], float)
    b = np.array([g2["region_counts"][r] for r in REGIONS], float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def competing_label(delta):
    if delta <= -2:
        return "DECREASED"
    if delta >= 2:
        return "INCREASED"
    return "ABOUT_THE_SAME" if delta == 0 else None   # |delta| == 1: declared ambiguous band


def true_class_share(state, label):
    """Share of the top-10 any-class match strength that belongs to the true class (continuous;
    PCA-W is ~binary per sample on ProtoViT because off-class last-layer weights are ~1e-5)."""
    acts = np.array([max(float(t["act"]), 0.0) for t in state["top_any"]])
    own = np.array([int(t["cls"]) == int(label) for t in state["top_any"]])
    return float(acts[own].sum() / acts.sum()) if acts.sum() > 0 else 0.0


def presentation_gt(pair, order):
    g1, g2 = pair["gt"][order[0]], pair["gt"][order[1]]
    s1, s2 = pair["states"][order[0]], pair["states"][order[1]]
    d1, d2 = dilate(g1["cells"]), dilate(g2["cells"])
    ph = (None if g1["part_hit_near"] is None or g2["part_hit_near"] is None
          else g2["part_hit_near"] - g1["part_hit_near"])
    delta = g2["n_competing"] - g1["n_competing"]
    return {
        "annotation_ok": pair["corruption"] not in ANNOTATION_UNRELIABLE,
        "prediction_changed": "YES" if g1["pred"] != g2["pred"] else "NO",
        "focus": [g1, g2],
        "spatial_distance": 1.0 - len(d1 & d2) / len(d1 | d2),       # no annotations needed
        "focus_tv": region_tv(g1, g2),                                # annotation-dependent
        "dominant_changed": "YES" if not set(g1["dominant"]) & set(g2["dominant"]) else "NO",
        "training_examples": examples_label(s1, s2),
        "delta_competing": delta, "competing_evidence": competing_label(delta),
        "d_correct": int(g2["correct"]) - int(g1["correct"]),
        "d_true_share": true_class_share(s2, pair["label"]) - true_class_share(s1, pair["label"]),
        "d_pcaw": g2["pcaw"] - g1["pcaw"],
        "d_background": g2["background_share"] - g1["background_share"],
        "d_part_hit": ph,
    }


def clear_state(g):
    return len(g["dominant"]) == 1 and g["dominant_share"] >= 0.5


# ----------------------------------------------------------------------------- statistics
def auc(scores, labels):
    """Mann-Whitney: P(score in the labelled group > score in the other group) + 0.5 P(tie)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y)[0])


def kappa(true, pred):
    from sklearn.metrics import cohen_kappa_score
    if len(true) == 0 or len(set(true) | set(pred)) < 2:
        return float("nan")
    return float(cohen_kappa_score(true, pred))


def best_constant(truth_sets):
    """Accuracy of the best constant answer (any option appearing in a truth set)."""
    opts = {o for s in truth_sets for o in s}
    return max(float(np.mean([o in s for s in truth_sets])) for o in opts) if truth_sets else float("nan")


def boot(rows, fn, seed=0):
    """Point estimate + stratified (by row['stratum']) cluster (by row['pair_id']) bootstrap."""
    est = fn(rows)
    out = {"est": est, "n": len(rows)}
    if not rows:
        return out
    rng = np.random.default_rng(seed)
    by_cluster, stratum_of = {}, {}
    for r in rows:
        by_cluster.setdefault(r["pair_id"], []).append(r)
        stratum_of[r["pair_id"]] = r["stratum"]
    by_stratum = {}
    for k, s in stratum_of.items():
        by_stratum.setdefault(s, []).append(k)
    vals = []
    for _ in range(N_BOOT):
        sample = []
        for keys in by_stratum.values():
            for i in rng.integers(0, len(keys), len(keys)):
                sample.extend(by_cluster[keys[i]])
        v = fn(sample)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    a = 100 * ALPHA / (2 * N_CLAIMS)
    if vals:
        lo, hi, flo, fhi = np.percentile(vals, [2.5, 97.5, a, 100 - a])
        out.update({"lo": float(lo), "hi": float(hi), "family_lo": float(flo), "family_hi": float(fhi)})
    out.update({"n_clusters": len(by_cluster), "n_valid_resamples": len(vals)})
    return out


def _count(xs):
    return dict(sorted(collections.Counter(xs).items()))


def _mean(f):
    return lambda rs: float(np.mean([f(r) for r in rs])) if rs else float("nan")


# ----------------------------------------------------------------------------- loading + provenance
def load_records(items_path, pattern, verify_stimuli=True):
    items_bytes = Path(items_path).read_bytes()
    doc = json.loads(items_bytes)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no record files match {pattern}")
    mans = {f: json.loads(Path(f[: -len(".jsonl")] + ".manifest.json").read_text()) for f in files}
    ref = mans[files[0]]
    for f, m in mans.items():
        for k in (set(ref) | set(m)) - PER_SHARD_KEYS:
            if m.get(k) != ref.get(k):
                raise SystemExit(f"manifest field {k!r} differs between shards ({f})")
    if ref.get("limit"):
        raise SystemExit("records come from a --limit (smoke) run; not analysable")
    if ref["items_sha256"] != sha256_bytes(items_bytes):
        raise SystemExit("records were produced from a different items file")
    if ref["full_plan_sha256"] != sha_json(doc["plan"]):
        raise SystemExit("items plan does not match the run's plan")
    n = ref["num_shards"]
    if sorted(m["shard_index"] for m in mans.values()) != list(range(n)):
        raise SystemExit(f"need exactly one file per shard 0..{n - 1}")
    index_of = {s["presentation_id"]: i for i, s in enumerate(doc["plan"])}
    pairs = {p["pair_id"]: p for p in doc["pairs"]}
    if verify_stimuli:
        import numpy as _np
        from narrator_run import render_stimuli
        bb = _np.load(PROTO_DIR / "bb4.npy")
    recs, stats = collections.defaultdict(list), {"records": 0}
    rendered = {}

    def pair_hashes(pid):
        if pid not in rendered:
            rendered[pid] = {kk: sha256_bytes(v) for kk, v in
                             render_stimuli(pairs[pid], doc["class_names"], bb).items()}
        return rendered[pid]

    for f in files:
        m = mans[f]
        k = m["shard_index"]
        plan_k = [s for i, s in enumerate(doc["plan"]) if i % n == k]
        if sha_json(plan_k) != m["plan_sha256"]:
            raise SystemExit(f"{f}: plan hash does not match shard {k} of the items plan")
        stim_hash = None
        if verify_stimuli:
            needed = sorted({s["pair_id"] for s in plan_k})
            stim_hash = {pid: pair_hashes(pid) for pid in needed}
            if sha_json(stim_hash) != m["stimuli_sha256"]:
                raise SystemExit(f"{f}: re-rendered stimuli do not match the manifest")
        own = sha_json(m)
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            stats["records"] += 1
            pid = r["presentation_id"]
            if r.get("manifest_sha256") != own:
                raise SystemExit(f"{pid}: record not bound to its own file's manifest")
            if pid not in index_of or index_of[pid] % n != k:
                raise SystemExit(f"{pid}: not a presentation of shard {k}")
            s = doc["plan"][index_of[pid]]
            pr = pairs[s["pair_id"]]
            if (r["pair_id"], r["condition"], r["order"], r["corruption"], r["stratum"]) != \
                    (s["pair_id"], s["condition"], s["order"], pr["corruption"], pr["stratum"]):
                raise SystemExit(f"{pid}: record metadata differ from the plan")
            if r["status"] == "ok" and stim_hash is not None:
                expect = [stim_hash[s["pair_id"]][kk] for kk in
                          ("test", f"{s['order'][0]}@1", f"{s['order'][1]}@2")]
                if r.get("image_sha256") != expect:
                    raise SystemExit(f"{pid}: shown images differ from the planned stimuli")
            recs[pid].append(r)
    final = {}
    for pid, rs in recs.items():   # file order; a presentation lives in exactly one shard file
        if any(r["status"] not in TERMINAL + ("error",) for r in rs):
            raise SystemExit(f"{pid}: unknown record status")
        term = [i for i, r in enumerate(rs) if r["status"] in TERMINAL]
        if len(term) > 1:
            raise SystemExit(f"{pid}: more than one terminal record (retry policy violated)")
        if term and term[0] != len(rs) - 1:
            raise SystemExit(f"{pid}: record after a terminal record (retry policy violated)")
        final[pid] = rs[-1]   # the terminal record, or the last infrastructure error (-> incomplete)
    missing = [pid for pid in index_of if final.get(pid, {}).get("status") not in TERMINAL]
    if missing:
        raise SystemExit(f"incomplete run: {len(missing)} presentations without a terminal record "
                         f"(e.g. {missing[:3]})")
    stats.update({"files": files, "num_shards": n, "stimuli_verified": verify_stimuli,
                  **{k: ref[k] for k in ("model_id", "enable_thinking", "sampling", "seed", "think_budget",
                                         "answer_budget", "early_stop_text", "code_sha256", "prompt_sha256")
                     if k in ref}})
    return doc, final, stats


# ----------------------------------------------------------------------------- analysis
def build_rows(doc, final):
    pairs = {p["pair_id"]: p for p in doc["pairs"]}
    rows = []
    for s in doc["plan"]:
        r = final[s["presentation_id"]]
        if r["status"] != "ok":
            continue
        pr = pairs[s["pair_id"]]
        rows.append({"pair_id": s["pair_id"], "condition": s["condition"], "stratum": pr["stratum"],
                     "corruption": pr["corruption"], "a": r["parsed"], "g": presentation_gt(pr, s["order"]),
                     "forced": any(at.get("forced_think_end") for at in r["attempts"])})
    return rows


def states_of(rows):
    return [{"pair_id": r["pair_id"], "stratum": r["stratum"], "annotation_ok": r["g"]["annotation_ok"],
             "ans": r["a"][f"main_focus_state{k + 1}"], "g": r["g"]["focus"][k]} for r in rows for k in (0, 1)]


def acc_vs_constant(rows, answer, truth_set):
    """(accuracy - best constant) with the constant re-chosen in every resample."""
    def fn(rs):
        if not rs:
            return float("nan")
        return float(np.mean([answer(r) in truth_set(r) for r in rs])) - best_constant([truth_set(r) for r in rs])
    return fn


def claim(rows, answer, truth_set):
    acc = boot(rows, _mean(lambda r: answer(r) in truth_set(r)))
    diff = boot(rows, acc_vs_constant(rows, answer, truth_set))
    return {"accuracy": acc, "minus_best_constant": diff,
            "best_constant": best_constant([truth_set(r) for r in rows]) if rows else None}


def primary(rows):
    ms = [r for r in rows if r["condition"] in ("main", "swapped")]
    ident = [r for r in rows if r["condition"] == "identical"]
    st = [s for s in states_of(ms) if s["annotation_ok"]]
    clear = [s for s in st if clear_state(s["g"])]
    chg = [r for r in ms if r["g"]["d_correct"] != 0]
    dec = [r for r in ms if r["g"]["competing_evidence"] is not None]
    V = {1: "IMPROVED", -1: "WORSENED"}

    claims = {
        "C1_prediction_changed": claim(ms, lambda r: r["a"]["prediction_changed"],
                                       lambda r: {r["g"]["prediction_changed"]}),
        "C2_main_focus_clear_states": claim(clear, lambda s: s["ans"], lambda s: set(s["g"]["dominant"])),
        "C4_training_examples": claim(ms, lambda r: r["a"]["training_examples"],
                                      lambda r: {r["g"]["training_examples"]}),
        "C5_competing_evidence_decisive": claim(dec, lambda r: r["a"]["competing_evidence"],
                                                lambda r: {r["g"]["competing_evidence"]}),
        "C6_verdict_direction_correctness_change": claim(chg, lambda r: r["a"]["verdict"],
                                                         lambda r: {V[r["g"]["d_correct"]]}),
    }
    claims["C3_focus_shift_spatial"] = {"auc": boot(ms, lambda rs: auc([r["g"]["spatial_distance"] for r in rs],
                                                                          [r["a"]["focus_shift"] == "YES" for r in rs]))}
    for c in claims.values():
        c["coverage"] = None
    claims["C2_main_focus_clear_states"]["coverage"] = {"clear_states": len(clear), "annotated_states": len(st)}
    claims["C5_competing_evidence_decisive"]["coverage"] = {"decisive": len(dec), "all": len(ms)}
    claims["C6_verdict_direction_correctness_change"]["coverage"] = {"correctness_changed": len(chg), "all": len(ms)}

    false_change = {
        "C1_prediction_changed": lambda a: a["prediction_changed"] == "YES",
        "C2_main_focus_clear_states": lambda a: a["main_focus_state1"] != a["main_focus_state2"],
        "C3_focus_shift_spatial": lambda a: a["focus_shift"] == "YES",
        "C4_training_examples": lambda a: a["training_examples"] != "SAME",
        "C5_competing_evidence_decisive": lambda a: a["competing_evidence"] != "ABOUT_THE_SAME",
        "C6_verdict_direction_correctness_change": lambda a: a["verdict"] != "NO_MEANINGFUL_CHANGE",
    }
    p2 = {k: boot(ident, (lambda f: _mean(lambda r: f(r["a"])))(f)) for k, f in false_change.items()}
    p2["any_field"] = boot(ident, _mean(lambda r: any(f(r["a"]) for f in false_change.values())))

    by_pair = {}
    for r in ms:
        by_pair.setdefault(r["pair_id"], {"stratum": r["stratum"]})[r["condition"]] = r["a"]
    both = [{"pair_id": k, "stratum": v["stratum"], "m": v["main"], "s": v["swapped"]}
            for k, v in by_pair.items() if "main" in v and "swapped" in v]
    swap = {
        "C1_prediction_changed": lambda m, s: m["prediction_changed"] == s["prediction_changed"],
        "C2_main_focus_clear_states": lambda m, s: (m["main_focus_state1"] == s["main_focus_state2"]
                                                    and m["main_focus_state2"] == s["main_focus_state1"]),
        "C3_focus_shift_spatial": lambda m, s: m["focus_shift"] == s["focus_shift"],
        "C4_training_examples": lambda m, s: m["training_examples"] == s["training_examples"],
        "C5_competing_evidence_decisive": lambda m, s: MIRROR["competing_evidence"][m["competing_evidence"]]
        == s["competing_evidence"],
        "C6_verdict_direction_correctness_change": lambda m, s: MIRROR["verdict"][m["verdict"]] == s["verdict"],
    }
    p3 = {k: boot(both, (lambda f: _mean(lambda x: f(x["m"], x["s"])))(f)) for k, f in swap.items()}
    p3["all_fields"] = boot(both, _mean(lambda x: all(f(x["m"], x["s"]) for f in swap.values())))

    labels = {}
    for k, c in claims.items():
        if k == "C3_focus_shift_spatial":
            e = c["auc"]
            above = e.get("family_lo") is not None and e["family_lo"] > 0.5
            strong = above and e["est"] >= RELIABLE_AUC
        else:
            e = c["minus_best_constant"]
            above = e.get("family_lo") is not None and e["family_lo"] > 0
            strong = above and c["accuracy"]["est"] >= RELIABLE_ACC
        ctrl = (p2[k]["est"] <= MAX_FALSE_CHANGE) and (p3[k]["est"] >= MIN_SWAP_CONSISTENCY)
        labels[k] = {"above_chance": bool(above), "accuracy_or_auc_threshold_met": bool(strong),
                     "controls_passed": bool(ctrl),
                     "label": "reliable" if (strong and ctrl) else ("above chance" if above else "not supported")}
    return {"claims": claims, "labels": labels, "P2_identical_false_change": p2, "P3_swap_consistency": p3,
            "n_pairs_both_orders": len(both)}


def secondary(rows):
    ms = [r for r in rows if r["condition"] in ("main", "swapped")]
    st_all = states_of(ms)
    st = [s for s in st_all if s["annotation_ok"]]
    unique = [s for s in st if len(s["g"]["dominant"]) == 1]
    ann = [r for r in ms if r["g"]["annotation_ok"]]
    same = [r for r in ms if r["g"]["d_correct"] == 0]
    v = lambda r: ORD["verdict"][r["a"]["verdict"]]
    focus_acc = _mean(lambda s: s["ans"] in s["g"]["dominant"])
    out = {
        "main_focus_all_annotated_states": boot(st, focus_acc),
        "main_focus_all_annotated_best_constant": best_constant([set(s["g"]["dominant"]) for s in st]) if st else None,
        "main_focus_elastic_only_approximate": boot([s for s in st_all if not s["annotation_ok"]], focus_acc),
        "main_focus_unique_kappa": boot(unique, lambda ss: kappa([s["g"]["dominant"][0] for s in ss],
                                                                 [s["ans"] for s in ss])),
        "main_focus_confusion_unique": {t: _count([s["ans"] for s in unique if s["g"]["dominant"][0] == t])
                                        for t in REGIONS},
        "background_answer_vs_background_share_auc": boot(
            st, lambda ss: auc([s["g"]["background_share"] for s in ss], [s["ans"] == "BACKGROUND" for s in ss])),
        "focus_shift_vs_region_tv_auc": boot(ann, lambda rs: auc([r["g"]["focus_tv"] for r in rs],
                                                                 [r["a"]["focus_shift"] == "YES" for r in rs])),
        "focus_shift_acc_dominant_changed": boot(ann, _mean(lambda r: r["a"]["focus_shift"] == r["g"]["dominant_changed"])),
        "training_examples_kappa": kappa([r["g"]["training_examples"] for r in ms],
                                         [r["a"]["training_examples"] for r in ms]),
        "training_examples_confusion": {t: _count([r["a"]["training_examples"] for r in ms
                                                   if r["g"]["training_examples"] == t])
                                        for t in ("SAME", "PARTLY", "DIFFERENT")},
        "competing_spearman_all": boot(ms, lambda rs: spearman([ORD["competing_evidence"][r["a"]["competing_evidence"]]
                                                                for r in rs], [r["g"]["delta_competing"] for r in rs])),
        "verdict_no_change_rate_on_correctness_change": boot(
            [r for r in ms if r["g"]["d_correct"] != 0], _mean(lambda r: v(r) == 0)),
        "verdict_auc_correctness_change": boot([r for r in ms if r["g"]["d_correct"] != 0],
                                               lambda rs: auc([v(r) for r in rs], [r["g"]["d_correct"] > 0 for r in rs])),
        "verdict_spearman_d_true_share_all": boot(ms, lambda rs: spearman([v(r) for r in rs],
                                                                          [r["g"]["d_true_share"] for r in rs])),
        "verdict_spearman_d_true_share_same_correctness": boot(
            same, lambda rs: spearman([v(r) for r in rs], [r["g"]["d_true_share"] for r in rs])),
        "verdict_spearman_neg_d_background_same_correctness_annotated": boot(
            [r for r in same if r["g"]["annotation_ok"]],
            lambda rs: spearman([v(r) for r in rs], [-r["g"]["d_background"] for r in rs])),
        "verdict_spearman_d_part_hit_same_correctness_annotated": boot(
            [r for r in same if r["g"]["annotation_ok"] and r["g"]["d_part_hit"] is not None],
            lambda rs: spearman([v(r) for r in rs], [r["g"]["d_part_hit"] for r in rs])),
        "verdict_spearman_d_pcaw_all_near_binary": boot(ms, lambda rs: spearman([v(r) for r in rs],
                                                                                [r["g"]["d_pcaw"] for r in rs])),
    }
    return out


def analyse(doc, final):
    rows = build_rows(doc, final)
    status = _count(final[s["presentation_id"]]["status"] for s in doc["plan"])
    rep = {"coverage": {"planned": len(doc["plan"]), "final_status": status, "analysed": len(rows),
                        "forced_think_end": int(sum(r["forced"] for r in rows))}}
    rep["primary"] = primary(rows)
    rep["secondary"] = secondary(rows)
    rep["sensitivity_excluding_forced_think_end"] = primary([r for r in rows if not r["forced"]])
    rep["by_stratum_main_accuracy"] = {
        s: {k: float(np.mean([r["a"][k] == r["g"][k] for r in rows if r["condition"] == "main" and r["stratum"] == s]))
            for k in ("prediction_changed", "training_examples")}
        for s in ("fixed", "broken", "both_right", "both_wrong")}
    w = doc["population_strata"]
    tot = sum(w.values())
    rep["stratum_reweighted_corruption_balanced_main_accuracy"] = {
        k: float(sum(w[s] / tot * rep["by_stratum_main_accuracy"][s][k] for s in w))
        for k in ("prediction_changed", "training_examples")}
    rep["answer_distributions"] = {c: {f: _count([r["a"][f] for r in rows if r["condition"] == c])
                                       for f in ("prediction_changed", "main_focus_state1", "main_focus_state2",
                                                 "focus_shift", "training_examples", "competing_evidence", "verdict")}
                                   for c in ("main", "swapped", "identical")}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--records", required=True, help="glob of shard .jsonl files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc, final, meta = load_records(args.items, args.records)
    rep = {"meta": meta, **analyse(doc, final)}
    Path(args.out).write_text(json.dumps(rep, indent=1, default=float))
    c = rep["primary"]["claims"]
    print(json.dumps({"coverage": rep["coverage"]}, indent=1, default=float))
    print(f"retrieved examples changed (accuracy): {c['C4_training_examples']['accuracy']['est']:.3f}")
    print(f"competing evidence rose/fell (accuracy): {c['C5_competing_evidence_decisive']['accuracy']['est']:.3f}")
    print(f"focus moved (AUC): {c['C3_focus_shift_spatial']['auc']['est']:.3f}")


if __name__ == "__main__":
    main()
