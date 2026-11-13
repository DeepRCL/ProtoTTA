#!/usr/bin/env python
"""Shared accounting for the blinded pairwise VLM judgments (vlm_judge.py / gemini_judge.py records).

Final outcome per (item, presentation) = last record in file order. Presentation score +1 (ProtoTTA),
-1 (EATA), 0 (same); item score = mean of the item's two presentation scores (both orders). ProtoTTA
share = P / (P + E) over decisive presentations, with an item-cluster bootstrap CI (items resampled,
both presentations kept together; 5,000 resamples, random.Random(20261107)).
"""
import json
from collections import Counter
from pathlib import Path

SCORE = {"prototta": 1, "eata": -1, "same": 0}
N_BOOT = 5000
SEED = 20261107


def load(files, item_ids):
    """Final record per (item, presentation) among the study items; all records counted."""
    recs, final = [], {}
    for f in files:
        p = Path(f)
        if not p.exists():
            return None
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["item_id"] not in item_ids:
                continue
            recs.append(r)
            final[(r["item_id"], r["presentation"])] = r   # file order: later record wins
    return recs, final


def boot_ci(stat, items, rng):
    vals = []
    for _ in range(N_BOOT):
        v = stat([rng.choice(items) for _ in items])
        if v == v:   # drop undefined resamples (nan)
            vals.append(v)
    vals.sort()
    if not vals:
        return [float("nan"), float("nan")]
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def summarize(final, item_ids, rng):
    """ProtoTTA share among decisive presentations (item-bootstrap CI) and item scores; an item
    missing an answered presentation gets no item score."""
    ok = {k: r for k, r in final.items() if r["status"] == "ok"}
    pres = {k: SCORE[r["choice_methods"]["better"]] for k, r in ok.items()}
    per_item = {}
    for (i, p), s in pres.items():
        per_item.setdefault(i, {})[p] = s
    complete = sorted(i for i, d in per_item.items() if len(d) == 2)
    item_score = {i: (per_item[i][0] + per_item[i][1]) / 2 for i in complete}

    def win(sample):
        P = sum(1 for i in sample for s in per_item.get(i, {}).values() if s > 0)
        E = sum(1 for i in sample for s in per_item.get(i, {}).values() if s < 0)
        return P / (P + E) if P + E else float("nan")

    c = Counter("P" if s > 0 else "E" if s < 0 else "same" for s in pres.values())
    return {
        "presentation_counts": dict(c),
        "win_rate": win(sorted(item_ids)), "win_rate_ci95": boot_ci(win, sorted(item_ids), rng),
        "item_scores": item_score,
    }


TERMINAL = ("ok", "parse_failed")   # judge statuses that end a presentation; "error" = re-run on resume


def completeness(final, item_ids):
    """Missing (item, presentation) keys and keys whose final record is not terminal."""
    expected = {(i, p) for i in item_ids for p in (0, 1)}
    missing = sorted(expected - set(final))
    pending = sorted(k for k in expected & set(final) if final[k]["status"] not in TERMINAL)
    return missing, pending
