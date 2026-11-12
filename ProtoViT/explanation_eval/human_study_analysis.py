#!/usr/bin/env python
"""Human study analysis (Table 5 left, human row; Appendix P).

Each participant judged the 16 items once: "Which explanation is better?" with A / B / "No clear
difference". A judgment is +1 (ProtoTTA's card chosen), -1 (EATA's card chosen) or 0 (no clear
difference), decoded with the release's private card key.
* ProtoTTA share = P / (P + E) over all judgments that preferred one method, on all items and on the
  items where both methods are correct (stratum both_right). 95% CI: two-way bootstrap that resamples
  participants and items independently (5,000 resamples, numpy default_rng(20261107)); resamples
  without a decisive judgment are dropped.
* Agreement with a VLM judge (--vlm_records, the judge's records on the same cards, both orders):
  human item score = mean judgment over participants; VLM item score = mean of its two presentation
  scores (pairwise_stats.py). Spearman correlation over the items with an item bootstrap CI (5,000
  resamples, numpy default_rng(20261107); resamples with a constant score vector are dropped), and
  the number of items with a non-tied human and VLM preference on which the two agree.

Input: the response export (CSV with columns received_at, participant_id, release, payload_json; the
payload holds the participant's responses), the release's card key and the final items file.
"""
import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairwise_stats as JA  # noqa: E402

SEED = 20261107
N_BOOT = 5000


def load_judgments(csv_path, key, release, exclude, received_before):
    rows = list(csv.DictReader(open(csv_path, newline="")))
    R = collections.defaultdict(dict)
    order = []
    for r in rows:
        pid = r["participant_id"]
        if r["release"] != release or any(pid.startswith(x) for x in exclude):
            continue
        if received_before and r["received_at"] > received_before:
            continue
        if pid in order:
            raise SystemExit(f"participant {pid[:8]} submitted twice; resolve before analysis")
        order.append(pid)
        for x in json.loads(r["payload_json"])["responses"]:
            if x.get("is_repeat"):
                continue
            b = x["better"]
            R[pid][x["item_id"]] = 0 if b not in key else (1 if key[b]["method"] == "prototta" else -1)
    return R, order


def counts(R, ps, its):
    p = e = s = 0
    for a in ps:
        for i in its:
            v = R[a].get(i)
            p += v == 1
            e += v == -1
            s += v == 0
    return p, e, s


def share_ci(R, order, its, seed=SEED, n=N_BOOT):
    """Two-way (participant x item) bootstrap of P / (P + E)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ps = [order[j] for j in rng.integers(0, len(order), len(order))]
        ii = [its[j] for j in rng.integers(0, len(its), len(its))]
        p, e, _ = counts(R, ps, ii)
        if p + e:
            vals.append(p / (p + e))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def rho_ci(h, v, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(0, len(h), len(h))
        if np.ptp(h[k]) == 0 or np.ptp(v[k]) == 0:
            continue
        out.append(spearmanr(h[k], v[k])[0])
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="response export (CSV)")
    ap.add_argument("--card_key", required=True, help="card_key_PRIVATE.json of the deployed release (render_cards.py)")
    ap.add_argument("--items", required=True, help="final items file of the human study")
    ap.add_argument("--release", default=None, help="release id (default: the card key's release)")
    ap.add_argument("--exclude", nargs="*", default=[], help="participant ids (or id prefixes) to exclude")
    ap.add_argument("--received_before", default=None, help="ISO-8601 timestamp; later submissions are ignored")
    ap.add_argument("--vlm_records", nargs="*", default=[], help="VLM judge records on the same cards (both orders)")
    ap.add_argument("--vlm_name", default="Gemini 3.8 Flash")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    key_doc = json.loads(Path(a.card_key).read_text())
    key = key_doc["cards"]
    release = a.release or key_doc["release"]
    items = {it["item_id"]: it for it in json.loads(Path(a.items).read_text())["items"]}
    R, order = load_judgments(a.responses, key, release, a.exclude, a.received_before)
    allit = sorted({i for p in R for i in R[p]})
    out = {"participants": len(order), "subsets": {}}
    print(f"participants {len(order)}")
    for name, its in (("all", allit), ("both_right", [i for i in allit if items[i]["stratum"] == "both_right"])):
        p, e, s = counts(R, order, its)
        lo, hi = share_ci(R, order, its)
        out["subsets"][name] = {"items": len(its), "ProtoTTA": p, "EATA": e, "same": s,
                                "share": p / (p + e), "ci95": [lo, hi]}
        print(f"{name}: items {len(its)}  ProtoTTA {p}  EATA {e}  no clear difference {s}  "
              f"share {100 * p / (p + e):.1f}  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]")
    if a.vlm_records:
        iids = sorted(items)
        got = JA.load(a.vlm_records, set(iids))
        if got is None:
            raise SystemExit("VLM record file missing")
        _, final = got
        missing, pending = JA.completeness(final, set(iids))
        if missing or pending:
            raise SystemExit(f"VLM run incomplete: {len(missing)} missing, {len(pending)} non-terminal")
        vs = JA.summarize(final, set(iids), random.Random(JA.SEED))["item_scores"]
        unscored = [i for i in iids if i not in vs]
        if unscored:
            raise SystemExit(f"VLM item score needs two answered presentations; missing for {unscored}")
        h = np.array([np.mean([R[p][i] for p in order if i in R[p]]) for i in iids])
        v = np.array([vs[i] for i in iids])
        rho = spearmanr(h, v)[0]
        lo, hi = rho_ci(h, v)
        hs, vsg = np.sign(h), np.sign(v)
        both = (hs != 0) & (vsg != 0)
        agree = int((hs[both] == vsg[both]).sum())
        out["agreement"] = {"judge": a.vlm_name, "n_items": len(iids), "spearman": float(rho),
                            "ci95": [float(lo), float(hi)], "both_decisive": int(both.sum()), "agree": agree}
        print(f"humans vs {a.vlm_name}: Spearman rho {rho:.2f} [{lo:.2f}, {hi:.2f}] over {len(iids)} images; "
              f"same preference on {agree} of {int(both.sum())} images with non-tied preferences")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
