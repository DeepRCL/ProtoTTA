#!/usr/bin/env python
"""Analysis of the narrator validation (Table 5, right; Appendix P).

Scores the narrator's fixed-vocabulary answers on the main and swapped presentations (the two
boards in both orders) against what the two boards show (the model's own records):
  retrieved training examples changed   accuracy of SAME / PARTLY / DIFFERENT (the five training-example
                                        pictures of the two boards: same multiset / none shared / partly)
  competing evidence rose / fell        accuracy of DECREASED / ABOUT_THE_SAME / INCREASED, on the pairs
                                        whose number of other-class entries among the ten strongest
                                        matches changes by at least 2 or not at all
  focus moved                           AUC of the narrator's YES / NO against the measured shift of the
                                        boxed cells (1 - Jaccard of the cells dilated by one cell)
Provenance: every shard's manifest, plan membership, record metadata and (by re-rendering) every
stimulus hash are checked; the run must be complete.
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
from render_cards import PROTO_DIR  # noqa: E402

TERMINAL = ("ok", "parse_failed")           # "error" (infrastructure) is re-run on resume
PER_SHARD_KEYS = {"shard_index", "plan_sha256", "stimuli_sha256"}


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


def competing_label(delta):
    if delta <= -2:
        return "DECREASED"
    if delta >= 2:
        return "INCREASED"
    return "ABOUT_THE_SAME" if delta == 0 else None   # |delta| == 1: declared ambiguous band


def presentation_gt(pair, order):
    g1, g2 = pair["gt"][order[0]], pair["gt"][order[1]]
    s1, s2 = pair["states"][order[0]], pair["states"][order[1]]
    d1, d2 = dilate(g1["cells"]), dilate(g2["cells"])
    delta = g2["n_competing"] - g1["n_competing"]
    return {
        "spatial_distance": 1.0 - len(d1 & d2) / len(d1 | d2),
        "training_examples": examples_label(s1, s2),
        "competing_evidence": competing_label(delta),
    }


# ----------------------------------------------------------------------------- statistics
def auc(scores, labels):
    """Mann-Whitney: P(score in the labelled group > score in the other group) + 0.5 P(tie)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


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
                     "corruption": pr["corruption"], "a": r["parsed"], "g": presentation_gt(pr, s["order"])})
    return rows


def measures(rows):
    """The three narrator measures of Table 5 (right), on the main and swapped presentations."""
    ms = [r for r in rows if r["condition"] in ("main", "swapped")]
    dec = [r for r in ms if r["g"]["competing_evidence"] is not None]
    return {
        "retrieved_examples_changed_accuracy": {
            "est": _mean(lambda r: r["a"]["training_examples"] in {r["g"]["training_examples"]})(ms), "n": len(ms)},
        "competing_evidence_accuracy": {
            "est": _mean(lambda r: r["a"]["competing_evidence"] in {r["g"]["competing_evidence"]})(dec), "n": len(dec)},
        "focus_moved_auc": {
            "est": auc([r["g"]["spatial_distance"] for r in ms], [r["a"]["focus_shift"] == "YES" for r in ms]),
            "n": len(ms)},
    }


def analyse(doc, final):
    rows = build_rows(doc, final)
    status = _count(final[s["presentation_id"]]["status"] for s in doc["plan"])
    return {"coverage": {"planned": len(doc["plan"]), "final_status": status, "analysed": len(rows)},
            "table5_right": measures(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--records", required=True, help="glob of shard .jsonl files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc, final, meta = load_records(args.items, args.records)
    rep = {"meta": meta, **analyse(doc, final)}
    Path(args.out).write_text(json.dumps(rep, indent=1, default=float))
    m = rep["table5_right"]
    print(json.dumps({"coverage": rep["coverage"]}, indent=1, default=float))
    print(f"retrieved examples changed (accuracy): {m['retrieved_examples_changed_accuracy']['est']:.3f}")
    print(f"competing evidence rose/fell (accuracy): {m['competing_evidence_accuracy']['est']:.3f}")
    print(f"focus moved (AUC): {m['focus_moved_auc']['est']:.3f}")


if __name__ == "__main__":
    main()
