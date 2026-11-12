#!/usr/bin/env python
"""Second VLM judge (Gemini 3.8 Flash) on the SAME stimuli, schedule, instructions and question as
vlm_judge.py (pairwise ProtoTTA vs EATA cards, both orders).

Transport: Gemini is called through an agentic command-line client (the Antigravity CLI, `agy`) that
is given the images as local files. The client must accept
    <cli> --model <name> --disable-slash-commands --sandbox --output-format stream-json -p <prompt>
run in a working directory that holds the images, offer a `read_file` tool, and print one JSON event
per line (tool steps as {"step_update": {"step_type": "tool", "state": ..., "tool_info": {"name": ...,
"parameters": {"Path": ...}}}}; the final answer as {"event": "result", "result": {"response": ...}}).
The client is found via --agent_cli, else $GEMINI_AGENT_CLI, else `agy` on PATH.
Each presentation runs in a fresh temporary directory under a neutral root outside the experiment
tree, containing exactly three images with neutral names (1_photo.jpg, 2_explanation_A.jpg,
3_explanation_B.jpg); the agent is told to open only those files.
Blinding is VERIFIED per attempt, not assumed: every tool call is logged and an answer is accepted
only if every successful tool call was `read_file` on one of the three images and all three were
read. Otherwise the attempt is marked "isolation_violation"/"missing_reads" and not counted.
Stimuli, schedule, prompt body and parser are imported from vlm_judge.py so both judges are directly
comparable. Runs on CPU; calls are parallelised with a thread pool. Same audit rules: every attempt
(prompt, tool calls, response, return code, stderr tail, seconds, status; timeouts with partial
output) is stored; preparation failures are persisted; a run manifest (incl. stimulus digest) must
match on resume.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_judge import (DESIGNS, RETRY_SUFFIX, build_schedule, choice_methods, instructions_for,  # noqa: E402
                      parse_answer, render_item_images, sha256_bytes)

AGY = os.environ.get("GEMINI_AGENT_CLI") or "agy"     # overridden by --agent_cli
PAIR_FILES = ("1_photo.jpg", "2_explanation_A.jpg", "3_explanation_B.jpg")
PAIR_FILE_PREAMBLE = (
    "Open and look at each of these three image files in the current directory with your read_file tool. "
    "Do not run any shell command and do not open any other file:\n"
    "1_photo.jpg = IMAGE 1; 2_explanation_A.jpg = IMAGE 2 (Explanation A); "
    "3_explanation_B.jpg = IMAGE 3 (Explanation B).\n\n"
)
TRANSPORT = {"pairwise": (PAIR_FILES, PAIR_FILE_PREAMBLE)}
NEUTRAL_ROOT = Path(tempfile.gettempdir()) / "vjudge_ws"   # outside the experiment/output tree; per-call image dirs


def _decode(x):
    return x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else (x or "")


def parse_stream(stdout, workdir, files=PAIR_FILES):
    """Return (response_text, tool_calls, blinding_status)."""
    response, calls = "", []
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        su = ev.get("step_update") or {}
        if su.get("step_type") == "tool" and su.get("state") in ("DONE", "ERROR"):
            info = su.get("tool_info") or {}
            calls.append({"name": info.get("name") or su.get("tool_name"), "state": su.get("state"),
                          "parameters": info.get("parameters")})
        if ev.get("event") == "result":
            response = (ev.get("result") or {}).get("response") or ""
    allowed = {str((workdir / f).resolve()) for f in files}
    read = set()
    for c in calls:
        if c["state"] != "DONE":
            continue   # denied/failed calls returned nothing to the model
        path = (c["parameters"] or {}).get("Path") if isinstance(c["parameters"], dict) else None
        if c["name"] != "read_file" or path is None:
            return response, calls, "isolation_violation"
        p = Path(path)
        canon = str((p if p.is_absolute() else workdir / p).resolve())   # relative paths are relative to the CLI cwd
        if canon not in allowed:
            return response, calls, "isolation_violation"
        read.add(canon)
    return response, calls, ("ok" if read == allowed else "missing_reads")


def set_agent_cli(path):
    global AGY
    AGY = path


def call_agy(model, prompt, shown_bytes, timeout, files=PAIR_FILES):
    NEUTRAL_ROOT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="w", dir=NEUTRAL_ROOT))
    t0 = time.time()
    try:
        assert len(files) == len(shown_bytes), (files, len(shown_bytes))
        for name, b in zip(files, shown_bytes):
            (tmp / name).write_bytes(b)
        try:
            proc = subprocess.run([AGY, "--model", model, "--disable-slash-commands", "--sandbox",
                                   "--output-format", "stream-json", "-p", prompt],
                                  cwd=tmp, capture_output=True, text=True, timeout=timeout)
            stdout, rc, err = proc.stdout, proc.returncode, proc.stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            stdout, rc, err = _decode(exc.stdout), "timeout", _decode(exc.stderr)[-2000:]
        response, calls, blind = parse_stream(stdout, tmp, files)
        return {"response": response, "tool_calls": calls, "blinding": blind, "returncode": rc,
                "stderr_tail": err, "seconds": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import numpy as np
    from render_cards import PROTO_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent_cli", default=None, help="agentic CLI executable (default: $GEMINI_AGENT_CLI or agy)")
    ap.add_argument("--card_background", choices=["clean"], default="clean")
    ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--design", choices=sorted(DESIGNS), default="pairwise")
    ap.add_argument("--schedule", choices=["both_orders"], default="both_orders")
    ap.add_argument("--seed", type=int, default=20261107)   # same default as vlm_judge.py -> same plan
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--terminal_failures", action="store_true",
                    help="on resume, never re-run a presentation that already has a record (screening runs: "
                         "a failure after the fixed two attempts is final)")
    ap.add_argument("--max_records_per_presentation", type=int, default=None,
                    help="cumulative cap across invocations: a presentation that already has this many records "
                         "(each record = up to two attempts) is never run again, whatever its status")
    ap.add_argument("--stop_file", default=None,
                    help="graceful stop: presentations not yet started when this file exists are skipped "
                         "without a record; in-flight presentations finish and are recorded")
    args = ap.parse_args()
    if args.agent_cli:
        set_agent_cli(args.agent_cli)

    design = DESIGNS[args.design]
    if args.schedule not in design["schedules"]:
        raise SystemExit(f"schedule {args.schedule} is not valid for design {args.design}")
    methods = design["methods"]
    files, preamble = TRANSPORT[args.design]
    items_bytes = Path(args.items).read_bytes()
    items_doc = json.loads(items_bytes)
    if items_doc.get("design") != args.design:
        raise SystemExit(f"items file is design {items_doc.get('design')!r}, not {args.design!r}")
    items = items_doc["items"]
    by_id = {it["item_id"]: it for it in items}
    full_plan = build_schedule(items, args.schedule, args.seed, methods)
    plan = full_plan[: args.limit] if args.limit else full_plan
    needed = sorted({s["item_id"] for s in plan})
    release = None
    bb = np.load(PROTO_DIR / "bb4.npy")
    stimuli, prep_failures = {}, []
    for i in needed:
        try:
            stimuli[i] = render_item_images(by_id[i], bb, args.card_background, methods)
        except Exception as exc:
            prep_failures.append({"item_id": i, "status": "prep_error", "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()[-4000:]})
    if prep_failures:
        fail_path = Path(args.out).with_suffix(".prep_failures.jsonl")
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("a") as fh:
            for r in prep_failures:
                fh.write(json.dumps(r) + "\n")
        raise SystemExit(f"{len(prep_failures)} stimulus preparation failures recorded in {fail_path}")
    stim_hash = {i: {k: sha256_bytes(v) for k, v in d.items()} for i, d in stimuli.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt1 = preamble + instructions_for(args.card_background, args.design)
    manifest = {
        "judge": "antigravity-cli", "model": args.model, "items_sha256": sha256_bytes(items_bytes),
        "release": release, "card_background": args.card_background, "schedule": args.schedule, "seed": args.seed,
        "full_plan_sha256": sha256_bytes(json.dumps(full_plan, sort_keys=True).encode()),
        "plan_sha256": sha256_bytes(json.dumps(plan, sort_keys=True).encode()),
        "stimuli_sha256": sha256_bytes(json.dumps(stim_hash, sort_keys=True).encode()),
        "prompt_sha256": sha256_bytes((prompt1 + RETRY_SUFFIX).encode()),
        "render_code_sha256": sha256_bytes((HERE / "render_cards.py").read_bytes()),
        "judge_code_sha256": sha256_bytes(Path(__file__).read_bytes()),   # acceptance policy is versioned
        "shared_code_sha256": sha256_bytes((HERE / "vlm_judge.py").read_bytes()),
        "design": args.design,
        **({"terminal_failures": True} if args.terminal_failures else {}),
        **({"max_records_per_presentation": args.max_records_per_presentation}
           if args.max_records_per_presentation else {}),
    }
    man_path = out.with_suffix(".manifest.json")
    if man_path.exists():
        if json.loads(man_path.read_text()) != manifest:
            raise SystemExit("Run manifest mismatch; refusing to resume.")
    else:
        if out.exists() and out.stat().st_size > 0:
            raise SystemExit(f"{out} has records but no manifest; refusing to mix runs")
        man_path.write_text(json.dumps(manifest, indent=2))
    done = set()
    n_records = Counter()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            n_records[(r["item_id"], r["presentation"])] += 1
            if args.max_records_per_presentation and \
                    n_records[(r["item_id"], r["presentation"])] >= args.max_records_per_presentation:
                done.add((r["item_id"], r["presentation"]))   # cumulative cap reached: final whatever its status
            if args.terminal_failures:
                done.add((r["item_id"], r["presentation"]))
            if r.get("status") == "ok":
                expect = [stim_hash[r["item_id"]]["test"]] + [stim_hash[r["item_id"]][m] for m in r["order"]]
                if r.get("image_sha256") != expect:
                    raise SystemExit(f"completed record {r['item_id']} used different stimuli")
                acc = r["attempts"][-1] if r.get("attempts") else {}
                if acc.get("returncode") != 0 or acc.get("blinding") != "ok" or not r.get("parsed"):
                    raise SystemExit(f"completed record {r['item_id']} p{r['presentation']} fails the acceptance policy")
                done.add((r["item_id"], r["presentation"]))

    lock = threading.Lock()

    def run_one(step):
        if args.stop_file and Path(args.stop_file).exists():   # graceful stop: not started -> no record
            print(f"[gemini] {step['item_id']} p{step['presentation']} skipped (stop file)", flush=True)
            return
        it = by_id[step["item_id"]]
        rec = {"item_id": it["item_id"], "corruption": it["corruption"], "sample_index": it["sample_index"],
               "presentation": step["presentation"], "order": step["order"], "manifest": manifest,
               "attempts": [], "status": "pending"}
        try:
            imgs = stimuli[it["item_id"]]
            shown = [imgs["test"]] + [imgs[m] for m in step["order"]]
            rec["image_sha256"] = [sha256_bytes(b) for b in shown]
            for attempt in (1, 2):
                prompt = prompt1 if attempt == 1 else prompt1 + RETRY_SUFFIX
                entry = {"attempt": attempt, "prompt": prompt, "raw": None, "returncode": None,
                         "stderr_tail": None, "seconds": None, "parse_status": "inference_error"}
                rec["attempts"].append(entry)
                res = call_agy(args.model, prompt, shown, args.timeout, files)
                answer, why = parse_answer(res["response"], require_think_end=False, design=args.design)
                if res["returncode"] != 0:
                    why, answer = f"returncode_{res['returncode']}", None   # failed/timed-out process never counts
                elif res["blinding"] != "ok":
                    why, answer = res["blinding"], None      # never count unverified-blind attempts
                entry.update({"raw": res["response"], "tool_calls": res["tool_calls"], "blinding": res["blinding"],
                              "returncode": res["returncode"], "stderr_tail": res["stderr_tail"],
                              "seconds": res["seconds"], "parse_status": why})
                if answer is not None:
                    rec.update({"status": "ok", "parsed": answer,
                                "choice_methods": choice_methods(answer, step["order"], args.design)})
                    break
            else:
                rec["status"] = "parse_failed"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-4000:]
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[gemini] {rec['item_id']} p{rec['presentation']} {rec['order']} -> {rec['status']} "
              f"{rec.get('choice_methods')}", flush=True)

    todo = [s for s in plan if (s["item_id"], s["presentation"]) not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(run_one, s) for s in todo]):
            f.result()


if __name__ == "__main__":
    main()
