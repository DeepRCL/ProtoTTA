#!/usr/bin/env python
"""Run the VLM narrator on the narrator-validation presentations.

The narrator sees IMAGE 1 = the damaged photo, IMAGE 2 = the STATE 1 board,
IMAGE 3 = the STATE 2 board (narrator_board.state_board) and a neutral prompt: it is
not told which state is adapted, which method produced it, the true class, or
whether either prediction is correct, and it is told the states may be identical.
It writes a free-text narration and answers fixed-vocabulary questions about the
boards; the answers are graded against the model internals by narrator_analysis.py.

Generation (Judge, think-budget handling, seeds) is reused from vlm_judge.py.
Auditing follows vlm_judge.py: every attempt is stored; a manifest (items sha256,
plan, prompt, model, generation settings, digest of every stimulus image) is written
on first start and must match on resume; completed records must carry the same
image hashes.
"""
import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PV = HERE.parent
sys.path.insert(0, str(HERE))

from narrator_items import REGIONS  # noqa: E402
from vlm_judge import EARLY_STOP_TEXT, THINK_END, sha256_bytes  # noqa: E402

YES_NO = ("YES", "NO")
FIELDS = {
    "prediction_changed": YES_NO,
    "main_focus_state1": REGIONS,
    "main_focus_state2": REGIONS,
    "focus_shift": YES_NO,
    "training_examples": ("SAME", "PARTLY", "DIFFERENT"),
    "competing_evidence": ("DECREASED", "ABOUT_THE_SAME", "INCREASED"),
    "verdict": ("IMPROVED", "WORSENED", "NO_MEANINGFUL_CHANGE"),
}
TEXT_FIELDS = ("state1_description", "state2_description", "change_summary")
RETRY_SUFFIX = "\nReturn ONLY the JSON object, with every key filled."
# When the thinking budget runs out, Qwen's early-stop text closes the reasoning; the narrator also
# opens its JSON answer so the model continues in the required format (otherwise forced stops can
# produce "key: value" lines). The prefix is part of the recorded raw text.
JSON_START = '{"state1_description": "'
NARRATOR_EARLY_STOP = EARLY_STOP_TEXT + JSON_START

PROMPT = """You are analysing the reasoning of an AI system that recognizes bird species. The system is a prototype network: it explains each decision by pointing at small square regions of the photo and matching each region to a region of a training example. You will compare two states of the same AI system (STATE 1 and STATE 2) on the same damaged photo.

IMAGE 1 is the damaged photo (noise, blur, fog, compression or similar) that the AI saw.
IMAGE 2 is the reasoning board of STATE 1. IMAGE 3 is the reasoning board of STATE 2.

How to read a board:
- The top line gives the class the AI predicted in that state.
- Below it, on the left, is the damaged photo with small coloured boxes marking the regions the AI matched to its five strongest prototypes of the predicted class. Boxes of one colour (and number) belong together.
- To the right are five training examples. Example k has a frame of the same colour as box group k, and its own boxes show the region the AI thinks looks alike.
- The bottom panel lists the ten strongest prototype matches from ANY class: the class each prototype belongs to and its match strength.

You are not told whether either prediction is correct, or how the two states were obtained. The two boards may be identical, similar or very different. Describe only what the boards show, and do not assume that either state is better.

Write:
- "state1_description": 1-2 sentences on the evidence STATE 1 uses (which bird parts or other regions, what the training examples look like, which classes compete).
- "state2_description": the same for STATE 2.
- "change_summary": 1-2 sentences on what changed from STATE 1 to STATE 2 (say so if nothing meaningful changed).

Then answer, using exactly one of the given options:
- "prediction_changed": "YES" if the two states predict different classes, otherwise "NO".
- "main_focus_state1" and "main_focus_state2": where most of the coloured boxes in that state's photo fall: "HEAD" (beak, eyes, crown, forehead, nape, throat), "BODY" (back, breast, belly), "WING", "TAIL", "LEGS", or "BACKGROUND" (not on the bird).
- "focus_shift": "YES" if the boxes in STATE 2 cover substantially different image regions than in STATE 1, otherwise "NO".
- "training_examples": "SAME" if STATE 2 shows the same five training-example pictures as STATE 1 (in any order), "DIFFERENT" if none of them is the same, "PARTLY" otherwise.
- "competing_evidence": compared with STATE 1, does the bottom panel of STATE 2 contain fewer, about the same number of, or more matches from classes other than that state's own predicted class? "DECREASED", "ABOUT_THE_SAME" or "INCREASED".
- "verdict": overall, is the reasoning of STATE 2 better, worse, or about as good as that of STATE 1? Good reasoning points at meaningful, discriminative parts of the bird, matches regions that look alike, and has consistent evidence for one class. "IMPROVED", "WORSENED" or "NO_MEANINGFUL_CHANGE".

Return exactly one JSON object and nothing else:
{"state1_description": "...", "state2_description": "...", "change_summary": "...", "prediction_changed": "YES|NO", "main_focus_state1": "HEAD|BODY|WING|TAIL|LEGS|BACKGROUND", "main_focus_state2": "HEAD|BODY|WING|TAIL|LEGS|BACKGROUND", "focus_shift": "YES|NO", "training_examples": "SAME|PARTLY|DIFFERENT", "competing_evidence": "DECREASED|ABOUT_THE_SAME|INCREASED", "verdict": "IMPROVED|WORSENED|NO_MEANINGFUL_CHANGE"}"""


def norm_option(v):
    return str(v).strip().upper().replace(" ", "_").replace("-", "_")


def parse_answer(text, require_think_end):
    """Last JSON object (after </think> if the model reasons) whose option fields are all valid
    and whose text fields are non-empty strings."""
    if THINK_END in text:
        answer_text = text.rsplit(THINK_END, 1)[1]
    elif require_think_end:
        return None, "no_think_end"
    else:
        answer_text = text
    dec = json.JSONDecoder()
    found = None
    for start in [i for i, ch in enumerate(answer_text) if ch == "{"]:
        try:
            obj, _ = dec.raw_decode(answer_text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        vals = {k: norm_option(obj.get(k, "")) for k in FIELDS}
        if not all(vals[k] in opts for k, opts in FIELDS.items()):
            continue
        texts = {k: obj.get(k) for k in TEXT_FIELDS}
        if not all(isinstance(t, str) and t.strip() for t in texts.values()):
            continue
        found = {**vals, **{k: t.strip()[:2000] for k, t in texts.items()}}
    return (found, "ok") if found else (None, "no_valid_json")


def render_stimuli(pair, class_names, bb):
    """{'test': jpeg bytes, '<method>@1': board as STATE 1, '<method>@2': board as STATE 2}."""
    from narrator_board import damaged_photo, state_board
    root = PV / pair["dataset_root"]
    out = {}

    def enc(im, q):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        return buf.getvalue()

    out["test"] = enc(damaged_photo(root, pair["path"]), 92)
    for m, st in pair["states"].items():
        for pos in (1, 2):
            out[f"{m}@{pos}"] = enc(state_board(root, pair["path"], st, class_names, bb, f"STATE {pos}"), 90)
    return out


def shown_keys(step):
    return ["test", f"{step['order'][0]}@1", f"{step['order'][1]}@2"]


def main():
    import numpy as np
    from PIL import Image
    from render_cards import PROTO_DIR
    from vlm_judge import Judge

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen3.6-35B-A3B")
    # Reported settings: thinking on (model default), recommended sampling seeded per call, think budget
    # 12,288, answer budget 2,048. max_new_tokens only applies when --think_budget 0 (unbudgeted).
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--think_budget", type=int, default=12288, help="0 = no budget (use --max_new_tokens)")
    ap.add_argument("--answer_budget", type=int, default=2048)
    ap.add_argument("--enable_thinking", choices=["default", "on", "off"], default="default")
    ap.add_argument("--sampling", choices=["greedy", "recommended"], default="recommended")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only (first N planned presentations)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = ap.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    if args.enable_thinking == "off" and args.think_budget:
        raise SystemExit("--enable_thinking off needs --think_budget 0")

    items_bytes = Path(args.items).read_bytes()
    doc = json.loads(items_bytes)
    by_id = {p["pair_id"]: p for p in doc["pairs"]}
    full_plan = doc["plan"]
    plan = [s for k, s in enumerate(full_plan) if k % args.num_shards == args.shard_index]
    if args.limit:
        plan = plan[: args.limit]
    bb = np.load(PROTO_DIR / "bb4.npy")
    needed = sorted({s["pair_id"] for s in plan})
    stimuli, prep_failures = {}, []
    for pid in needed:   # all rendered up front; failures are persisted, then the run stops
        try:
            stimuli[pid] = render_stimuli(by_id[pid], doc["class_names"], bb)
        except Exception as exc:  # noqa: BLE001
            prep_failures.append({"pair_id": pid, "status": "prep_error", "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()[-4000:]})
    if prep_failures:
        fail_path = Path(args.out).with_suffix(".prep_failures.jsonl")
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("a") as fh:
            for r in prep_failures:
                fh.write(json.dumps(r) + "\n")
        raise SystemExit(f"{len(prep_failures)} stimulus preparation failures recorded in {fail_path}")
    stim_hash = {pid: {k: sha256_bytes(v) for k, v in d.items()} for pid, d in stimuli.items()}
    require_think_end = args.enable_thinking != "off" and ("Thinking" in args.model_id or "Qwen3.6" in args.model_id)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "items_sha256": sha256_bytes(items_bytes), "items_kind": doc.get("kind"),
        "full_plan_sha256": sha256_bytes(json.dumps(full_plan, sort_keys=True).encode()),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "plan_sha256": sha256_bytes(json.dumps(plan, sort_keys=True).encode()),
        "stimuli_sha256": sha256_bytes(json.dumps(stim_hash, sort_keys=True).encode()),
        "prompt_sha256": sha256_bytes((PROMPT + RETRY_SUFFIX).encode()), "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "require_think_end": require_think_end, "sampling": args.sampling, "seed": args.seed,
        **({"think_budget": args.think_budget, "answer_budget": args.answer_budget,
            "early_stop_text": NARRATOR_EARLY_STOP} if args.think_budget else {}),
        "code_sha256": {f: sha256_bytes((HERE / f).read_bytes())
                        for f in ("narrator_run.py", "narrator_board.py", "render_cards.py", "vlm_judge.py")},
        "limit": args.limit,
    }
    man_path = out.with_suffix(".manifest.json")
    if man_path.exists():
        old = json.loads(man_path.read_text())
        if old != manifest:
            raise SystemExit(f"Run manifest mismatch; refusing to resume.\nold={old}\nnew={manifest}")
    else:
        if out.exists() and out.stat().st_size > 0:
            raise SystemExit(f"{out} has records but no manifest; refusing to mix runs")
        man_path.write_text(json.dumps(manifest, indent=2))
    # Terminal records ("ok", or "parse_failed" after the one allowed retry) are never re-run;
    # "error" records (infrastructure exceptions) are re-run on resume.
    manifest_hash = sha256_bytes(json.dumps(manifest, sort_keys=True).encode())
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("manifest_sha256") != manifest_hash:
                raise SystemExit(f"record {r.get('presentation_id')} belongs to a different manifest")
            if r.get("status") in ("ok", "parse_failed"):
                step = {"order": r["order"]}
                if r.get("image_sha256") != [stim_hash[r["pair_id"]][k] for k in shown_keys(step)]:
                    raise SystemExit(f"completed record {r['presentation_id']} used different stimuli")
                done.add(r["presentation_id"])
    think = {"default": None, "on": True, "off": False}[args.enable_thinking]
    judge = Judge(args.model_id, args.max_new_tokens, think, args.think_budget or None, args.answer_budget,
                  args.sampling, early_stop_text=NARRATOR_EARLY_STOP)
    with out.open("a") as fh:
        for step in plan:
            if step["presentation_id"] in done:
                continue
            pr = by_id[step["pair_id"]]
            rec = {"presentation_id": step["presentation_id"], "pair_id": step["pair_id"],
                   "condition": step["condition"], "order": step["order"], "corruption": pr["corruption"],
                   "stratum": pr["stratum"], "manifest_sha256": manifest_hash,
                   "attempts": [], "status": "pending"}
            try:
                imgs = stimuli[step["pair_id"]]
                keys = shown_keys(step)
                rec["image_sha256"] = [sha256_bytes(imgs[k]) for k in keys]
                pil = [Image.open(io.BytesIO(imgs[k])).convert("RGB") for k in keys]
                for attempt in (1, 2):
                    prompt = PROMPT if attempt == 1 else PROMPT + RETRY_SUFFIX
                    entry = {"attempt": attempt, "raw": None, "new_tokens": None, "seconds": None,
                             "parse_status": "inference_error"}
                    rec["attempts"].append(entry)        # recorded before inference
                    t0 = time.time()
                    call_seed = int(sha256_bytes(f"{args.seed}:{step['presentation_id']}:{attempt}".encode())[:8], 16)
                    entry["seed"] = call_seed
                    try:
                        raw, n_new, gen_info = judge(pil, prompt, call_seed)
                    except Exception as exc:
                        entry.update({"seconds": round(time.time() - t0, 1),
                                      "error": f"{type(exc).__name__}: {exc}"})
                        raise
                    answer, why = parse_answer(raw, require_think_end)
                    entry.update({"raw": raw, "new_tokens": n_new, **gen_info,
                                  "seconds": round(time.time() - t0, 1), "parse_status": why})
                    if answer is not None:
                        rec.update({"status": "ok", "parsed": answer})
                        break
                else:
                    rec["status"] = "parse_failed"
            except Exception as exc:  # noqa: BLE001  (recorded, never dropped)
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()[-4000:]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            p = rec.get("parsed", {})
            print(f"[narrator] {rec['presentation_id']} -> {rec['status']} "
                  f"{ {k: p.get(k) for k in FIELDS} if p else ''}", flush=True)


if __name__ == "__main__":
    main()
