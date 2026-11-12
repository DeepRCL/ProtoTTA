#!/usr/bin/env python
"""Blinded pairwise VLM comparison on 104 images, ProtoTTA vs EATA (Table 5 left, VLM rows; Appendix P).

A judge is analysed only if its run is complete: every (item, presentation 0/1) has a final record
with a terminal status (ok or parse_failed). Per judge: ProtoTTA share among decisive presentations
(both orders of every item) with an item-cluster percentile-bootstrap 95% CI (pairwise_stats.py), on
all 104 images and on the images where both methods are correct (stratum both_right).

  python vlm104_analysis.py --items items_pairwise104.json \
      --judge "Gemini 3.8 Flash" gemini.jsonl --judge "Qwen3.6-35B-A3B" qwen_s0.jsonl qwen_s1.jsonl qwen_s2.jsonl \
      --out vlm104_analysis.json
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairwise_stats as JA  # noqa: E402

SUBSETS = ("all", "both_right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="make_items_vlm104.py output")
    ap.add_argument("--judge", nargs="+", action="append", required=True, metavar=("NAME", "RECORDS"),
                    help="judge name followed by its record file(s) (all shards); repeat per judge")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = json.loads(Path(a.items).read_text())
    stratum = {it["item_id"]: it["stratum"] for it in doc["items"]}
    all_ids = set(stratum)
    subsets = {"all": all_ids, "both_right": {i for i, s in stratum.items() if s == "both_right"}}
    out = {"n_items": len(all_ids), "stratum_counts": dict(Counter(stratum.values())), "judges": {}}
    for spec in a.judge:
        if len(spec) < 2:
            raise SystemExit(f"--judge needs a name and at least one record file: {spec}")
        judge, files = spec[0], spec[1:]
        got = JA.load(files, all_ids)
        if got is None:
            raise SystemExit(f"{judge}: record file missing")
        _, final = got
        missing, pending = JA.completeness(final, all_ids)
        if missing or pending:
            print(f"[skip] {judge}: incomplete run ({len(missing)} presentations missing, {len(pending)} without a "
                  f"terminal record, e.g. {(missing + pending)[:3]}); resume it before analysis")
            continue
        res = {"subsets": {}}
        for name in SUBSETS:
            ids = subsets[name]
            sub = {k: r for k, r in final.items() if k[0] in ids}
            s = JA.summarize(sub, ids, random.Random(JA.SEED))
            res["subsets"][name] = {"primary": s}
            ci = s["win_rate_ci95"]
            print(f"{judge:24s} {name:10s} n={len(ids):3d} presentations {s['presentation_counts']}  ProtoTTA share "
                  f"{100 * s['win_rate']:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
        out["judges"][judge] = res
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
