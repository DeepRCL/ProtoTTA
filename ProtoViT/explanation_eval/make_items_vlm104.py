#!/usr/bin/env python
"""Items for the 104-photo blinded pairwise VLM comparison (ProtoTTA vs EATA).

Rule (label-free): walk the seeded, label-free order of the item pool written by
select_items.py --mode vlm_large (20 per corruption; adaptive ProtoTTA, EATA and unadapted states).
Within each corruption, in file order, skip an item if (a) its ProtoTTA and EATA cards are identical
as displayed (select_items.card_signature: same prototypes with the same active sub-patch boxes; the
pairwise human study used the same rule) or (b) its source photo is used in the pairwise human study;
keep the first 8. Correctness is stored for the offline analysis only and never shown.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_items import card_signature  # noqa: E402

PER_CORRUPTION = 8
STRATA = {(True, True): "both_right", (False, False): "both_wrong",
          (True, False): "prototta_right_eata_wrong", (False, True): "eata_right_prototta_wrong"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="select_items.py --mode vlm_large output")
    ap.add_argument("--human_items", required=True, help="final items of the pairwise human study")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    SRC, HUMAN_PAIRWISE, OUT = Path(args.source), Path(args.human_items), Path(args.out)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    src = json.loads(SRC.read_text())
    human_photos = {it["path"] for it in json.loads(HUMAN_PAIRWISE.read_text())["items"]}
    by_c = {}
    for it in src["items"]:
        by_c.setdefault(it["corruption"], []).append(it)
    items, skip = [], {}
    for c in sorted(by_c):
        skip[c] = {"identical_cards": 0, "in_human_pairwise_study": 0, "not_reached": 0}
        kept = 0
        for it in by_c[c]:
            if kept == PER_CORRUPTION:
                skip[c]["not_reached"] += 1
                continue
            if card_signature(it["methods"]["prototta"]) == card_signature(it["methods"]["eata"]):
                skip[c]["identical_cards"] += 1
                continue
            if it["path"] in human_photos:
                skip[c]["in_human_pairwise_study"] += 1
                continue
            p, e = it["methods"]["prototta"]["correct"], it["methods"]["eata"]["correct"]
            items.append({**it, "stratum": STRATA[(bool(p), bool(e))]})
            kept += 1
        if kept != PER_CORRUPTION:
            raise SystemExit(f"{c}: only {kept} eligible items in the pre-drawn order")
    doc = {
        "design": "pairwise", "methods_compared": ["eata", "prototta"], "n_items": len(items),
        "rule": __doc__, "source": str(SRC), "source_sha256": sha256(SRC),
        "human_pairwise_sha256": sha256(HUMAN_PAIRWISE), "per_corruption": PER_CORRUPTION,
        "skip_log": skip, "stratum_counts": dict(Counter(it["stratum"] for it in items)),
        "items": items,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT}: {len(items)} items; strata {doc['stratum_counts']}")
    print("skips:", {c: v for c, v in skip.items() if v['identical_cards'] or v['in_human_pairwise_study']})


if __name__ == "__main__":
    main()
