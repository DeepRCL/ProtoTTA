#!/usr/bin/env python
"""Screening rounds and final item list for the pairwise study (rule in select_pairwise.py).

Screening judge: Gemini (gemini_judge.py) in the human-study format (clean background,
--design pairwise, --schedule both_orders, rendered stimuli, --terminal_failures).
A candidate PASSES when both presentations (0 and 1, opposite method orders) have a
valid, blinding-verified answer; it FAILS when both presentations have a terminal record
and at least one is not ok. Each (candidate, presentation) may have exactly one terminal
record across all runs (no re-tries beyond the judge's fixed two attempts), so a failure
is final. Passing depends only on the judge returning a valid blinded answer, never on
which method it chose. Every run must be supplied with the round items file it used;
manifests, item contents and records are validated before anything is counted.

  round  --rounds r1.json ... --runs r1.jsonl ...  -> next round items file (round 1: no args)
  final  --rounds ... --runs ...                    -> items_pairwise.json (+ replacement log,
                                                       accepted record references)
  check_release --final items_pairwise.json --release_dir D
                                                    -> re-validates the screening rounds/runs recorded
                                                       in the final file; released card/test bytes must
                                                       equal the images Gemini judged in screening
"""
import argparse
import hashlib
import json
from pathlib import Path

N_PRESENTATIONS = 2
SCREEN = {"judge": "antigravity-cli", "model": "gemini-3.8-flash-high", "design": "pairwise",
          "card_background": "clean", "schedule": "both_orders", "release": None}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def load_status(cands, round_files, run_files):
    if len(round_files) != len(run_files):
        raise SystemExit("--rounds and --runs must be paired (same length, same order)")
    cand_by_id = {c["item_id"]: c for sl in cands["slots"] for c in sl["candidates"]}
    methods = set(cands["methods_compared"])
    terminal = {}   # (item_id, presentation) -> record ref
    for rf, run in zip(round_files, run_files):
        rbytes = Path(rf).read_bytes()
        rdoc = json.loads(rbytes)
        for it in rdoc["items"]:
            if cand_by_id.get(it["item_id"]) != it:
                raise SystemExit(f"{rf}: item {it['item_id']} differs from candidates.json")
        man = json.loads(Path(run).with_suffix(".manifest.json").read_text())
        for k, v in SCREEN.items():
            if man.get(k) != v:
                raise SystemExit(f"{run}: manifest {k}={man.get(k)!r}, screening requires {v!r}")
        if man.get("items_sha256") != sha256(rbytes):
            raise SystemExit(f"{run}: was not run on {rf}")
        if man.get("plan_sha256") != man.get("full_plan_sha256"):
            raise SystemExit(f"{run}: partial plan (--limit) is not a screening run")
        round_ids = {it["item_id"] for it in rdoc["items"]}
        for ln, line in enumerate(Path(run).read_text().splitlines(), 1):
            r = json.loads(line)
            if r.get("manifest") != man:
                raise SystemExit(f"{run}:{ln}: record manifest differs from the run manifest")
            if r["item_id"] not in round_ids or r["presentation"] not in range(N_PRESENTATIONS):
                raise SystemExit(f"{run}:{ln}: unexpected item/presentation")
            if set(r["order"]) != methods or len(r["order"]) != 2:
                raise SystemExit(f"{run}:{ln}: bad order {r['order']}")
            key = (r["item_id"], r["presentation"])
            if key in terminal:
                raise SystemExit(f"{run}:{ln}: second terminal record for {key} (re-tries are not allowed)")
            ok = False
            if r["status"] == "ok":
                acc = r["attempts"][-1]
                ok = (acc.get("returncode") == 0 and acc.get("blinding") == "ok"
                      and isinstance(r.get("parsed"), dict) and r["parsed"].get("better") in ("A", "B", "SAME"))
                if not ok:
                    raise SystemExit(f"{run}:{ln}: 'ok' record fails the acceptance policy")
            terminal[key] = {"run": run, "line": ln, "ok": ok, "order": r["order"],
                             "image_sha256": r.get("image_sha256")}
    for iid in {k[0] for k in terminal}:   # both presentations must use opposite orders
        recs = [terminal.get((iid, p)) for p in range(N_PRESENTATIONS)]
        if all(recs) and recs[0]["order"] != recs[1]["order"][::-1]:
            raise SystemExit(f"{iid}: presentations are not opposite orders")
    return terminal


def state(cand_id, terminal):
    recs = [terminal.get((cand_id, p)) for p in range(N_PRESENTATIONS)]
    if not any(recs):
        return "untried"
    if not all(recs):
        return "incomplete"
    return "pass" if all(r["ok"] for r in recs) else "fail"


def items_doc(cands, items, note):
    return {"design": "pairwise", "methods_compared": cands["methods_compared"], "seed": cands["seed"],
            "rule": cands["rule"], "note": note, "items": items, "repeat_item_ids": [],
            "tutorial": cands["tutorial"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["round", "final", "check_release"])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--rounds", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--release_dir", default=None)
    args = ap.parse_args()
    cands = json.loads(Path(args.candidates).read_text())
    final = None
    if args.cmd == "check_release":
        if args.rounds or args.runs:
            raise SystemExit("check_release uses the screening provenance recorded in --final")
        final = json.loads(Path(args.final).read_text())
        args.rounds, args.runs = final["screening"]["rounds"], final["screening"]["runs"]
    terminal = load_status(cands, args.rounds, args.runs)
    if args.out and Path(args.out).exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")

    if args.cmd == "round":
        items = []
        for sl in cands["slots"]:
            states = [state(c["item_id"], terminal) for c in sl["candidates"]]
            if "pass" in states:
                continue
            if "incomplete" in states:
                raise SystemExit(f"{sl['slot']}: a run is incomplete; finish it first")
            k = next((j for j, s in enumerate(states) if s == "untried"), None)
            if k is None:
                raise SystemExit(f"{sl['slot']}: all candidates failed")
            assert all(s == "fail" for s in states[:k])
            items.append(sl["candidates"][k])
        if not items:
            raise SystemExit("every slot already has a passing candidate; run 'final'")
        Path(args.out).write_text(json.dumps(items_doc(cands, items, "screening round"), indent=2))
        print(f"{len(items)} candidates to screen -> {args.out}")
    elif args.cmd == "final":
        items, log = [], []
        for sl in cands["slots"]:
            states = [state(c["item_id"], terminal) for c in sl["candidates"]]
            if "pass" not in states:
                raise SystemExit(f"{sl['slot']}: no passing candidate yet {states}")
            k = states.index("pass")
            if any(s != "fail" for s in states[:k]):
                raise SystemExit(f"{sl['slot']}: an earlier candidate has not failed {states}")
            cid = sl["candidates"][k]["item_id"]
            items.append(sl["candidates"][k])
            log.append({"slot": sl["slot"], "corruption": sl["corruption"], "stratum": sl["stratum"],
                        "chosen": cid, "n_replaced": k, "states": states,
                        "accepted_records": [{kk: terminal[(cid, p)][kk] for kk in ("run", "line")}
                                             for p in range(N_PRESENTATIONS)]})
        doc = items_doc(cands, items, "final items after Gemini screening")
        doc["replacement_log"] = log
        doc["screening"] = {"rounds": args.rounds, "runs": args.runs, "config": SCREEN}
        Path(args.out).write_text(json.dumps(doc, indent=2))
        print(json.dumps([(x["slot"], x["chosen"], x["n_replaced"]) for x in log]))
    else:   # check_release
        for x in final["replacement_log"]:   # the recorded accepted records are still the terminal ones
            for p, ref in enumerate(x["accepted_records"]):
                t = terminal[(x["chosen"], p)]
                if not t["ok"] or {"run": t["run"], "line": t["line"]} != ref:
                    raise SystemExit(f"{x['chosen']} p{p}: accepted record reference does not match")
        rel = Path(args.release_dir)
        key = json.loads((rel / "card_key_PRIVATE.json").read_text())
        if key["items_sha256"] != sha256(Path(args.final).read_bytes()):
            raise SystemExit("release was not built from this final items file")
        cfg = json.loads((rel / "site" / "study_config.json").read_text())
        assert cfg["design"] == "pairwise" and cfg["card_background"] == "clean"
        by_item = {it["item_id"]: it for it in cfg["items"]}
        assert set(by_item) == {it["item_id"] for it in final["items"]}
        n = 0
        for it in final["items"]:
            c = by_item[it["item_id"]]
            img = {"test": sha256((rel / "site" / "cards" / f"{c['test']}.jpg").read_bytes())}
            for cid in c["cards"]:
                img[key["cards"][cid]["method"]] = sha256((rel / "site" / "cards" / f"{cid}.jpg").read_bytes())
            for p in range(N_PRESENTATIONS):
                t = terminal[(it["item_id"], p)]
                expect = [img["test"]] + [img[m] for m in t["order"]]
                if t["image_sha256"] != expect:
                    raise SystemExit(f"{it['item_id']} p{p}: released images differ from the screened ones")
                n += 1
        print(f"release matches screening stimuli for {n} presentations")


if __name__ == "__main__":
    main()
