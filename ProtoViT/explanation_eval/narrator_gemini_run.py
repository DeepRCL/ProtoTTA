#!/usr/bin/env python
"""Second narrator: Gemini 3.8 Flash on the SAME narrator-validation presentations, boards, prompt
body and parser as narrator_run.py, called through the agentic CLI described in gemini_judge.py.

Differences from narrator_run.py are transport only:
* the three images are written with neutral names (1_photo.jpg, 2_board.jpg, 3_board.jpg) into a
  fresh temporary directory under a neutral root outside the experiment tree; a short preamble asks
  the model to open exactly these files, not to search the web or run commands, and to answer only
  from the images; the prompt body is narrator_run.PROMPT unchanged;
* the CLI runs with --agent blind-reader, an agent whose only tool is read_file (no web search, no
  shell); its definition is shipped in gemini_agent/blind-reader/ (install it as a plugin of the
  CLI) and hashed into the manifest;
* blinding is VERIFIED per attempt (gemini_judge.parse_stream): every successful tool call must be
  read_file on one of the three images and all three must be read; otherwise the attempt is
  rejected and repeated with the same prompt;
* the provider exposes no seed or temperature: sampling is the provider default (not reproducible
  call by call); the thinking level is fixed by the model name (e.g. gemini-3.8-flash-high).
Retry policy (mirrors narrator_run.py): at most two blinded, completed answers per presentation
(the second with narrator_run.RETRY_SUFFIX); if neither parses, the presentation is
"parse_failed" (terminal, never re-run). Transport failures (non-zero exit, timeout, blinding
rejection) are repeated with the same prompt up to --max_attempts calls in total; if no terminal
outcome is reached the record is "error" and is re-run on resume. The two-answer budget holds
ACROSS resumes (answers already given in earlier "error" records of the presentation are counted,
and the next answer gets the retry suffix if one was given); the transport-call cap applies per
invocation. Every call is journalled (<out>.journal.jsonl, fsync'd 'start' before and 'end' after
the call); on resume, answers spent in interrupted invocations are counted from the journal, a
terminal outcome that was reached but not written is recovered without a new call, and a call that
started but never finished (output never observed) blocks the resume unless --accept_interrupted
(then it counts as an unanswered transport failure; recorded per presentation).
Output: one shard (num_shards = 1) in the narrator_run.py record/manifest format, so
narrator_analysis.py reads it unchanged (stimuli are re-rendered and hash-checked there).
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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemini_judge as gj  # noqa: E402
from gemini_judge import NEUTRAL_ROOT, _decode, parse_stream  # noqa: E402
from narrator_run import FIELDS, PROMPT, RETRY_SUFFIX, parse_answer, render_stimuli, shown_keys  # noqa: E402
from vlm_judge import sha256_bytes  # noqa: E402

FILES = ("1_photo.jpg", "2_board.jpg", "3_board.jpg")
PREAMBLE = (
    "Open and look at each of these three image files in the current directory with your read_file tool. "
    "Do not search the web, do not run any shell command and do not open any other file; answer only "
    "from what the three images show:\n"
    "1_photo.jpg = IMAGE 1; 2_board.jpg = IMAGE 2; 3_board.jpg = IMAGE 3.\n\n"
)
AGENT_DIR = HERE / "gemini_agent" / "blind-reader"     # overridden by --agent_definition_dir
AGENT_FILES = ("plugin.json", "agents/blind-reader.md")


def call_agy(model, agent, prompt, shown_bytes, timeout):
    """One CLI call in a fresh neutral directory holding exactly the three images."""
    NEUTRAL_ROOT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="n", dir=NEUTRAL_ROOT))
    t0 = time.time()
    try:
        for name, b in zip(FILES, shown_bytes):
            (tmp / name).write_bytes(b)
        try:
            proc = subprocess.run([gj.AGY, "--agent", agent, "--model", model, "--disable-slash-commands", "--sandbox",
                                   "--output-format", "stream-json", "-p", prompt],
                                  cwd=tmp, capture_output=True, text=True, timeout=timeout)
            stdout, rc, err = proc.stdout, proc.returncode, proc.stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            stdout, rc, err = _decode(exc.stdout), "timeout", _decode(exc.stderr)[-2000:]
        response, calls, blind = parse_stream(stdout, tmp, FILES)
        return {"response": response, "tool_calls": calls, "blinding": blind, "returncode": rc,
                "stderr_tail": err, "seconds": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TRANSPORT = call_agy   # replaceable (e.g. by a fake transport for offline pipeline checks)


class Journal:
    """Append-only, fsync'd log of every call ('start' before it, 'end' after it), so that answers
    already spent and terminal outcomes survive an interrupted invocation."""

    def __init__(self, path, manifest_hash, invocation):
        self.mh, self.inv = manifest_hash, invocation
        self.lock = threading.Lock()
        self.fh = path.open("a")

    def log(self, **ev):
        ev.update(manifest_sha256=self.mh, invocation=self.inv, t=round(time.time(), 3))
        with self.lock:
            self.fh.write(json.dumps(ev) + "\n")
            self.fh.flush()
            os.fsync(self.fh.fileno())


def run_presentation(step, pr, imgs, manifest_hash, args, journal, prior_answers=0, interrupted_before=0):
    pid = step["presentation_id"]
    rec = {"presentation_id": pid, "pair_id": step["pair_id"],
           "condition": step["condition"], "order": step["order"], "corruption": pr["corruption"],
           "stratum": pr["stratum"], "manifest_sha256": manifest_hash, "attempts": [], "status": "pending",
           "invocation": journal.inv, "prior_answers": prior_answers,
           "interrupted_calls_before": interrupted_before}
    try:
        assert 0 <= prior_answers < 2, prior_answers
        keys = shown_keys(step)
        shown = [imgs[k] for k in keys]
        rec["image_sha256"] = [sha256_bytes(b) for b in shown]
        answered = prior_answers          # blinded, completed answers (each one may or may not parse)
        for call in range(1, args.max_attempts + 1):
            prompt = PREAMBLE + PROMPT + (RETRY_SUFFIX if answered == 1 else "")
            entry = {"call": call, "answer_attempt": answered + 1, "retry_suffix": answered == 1,
                     "counted_answer": False, "forced_think_end": False}
            rec["attempts"].append(entry)
            journal.log(type="start", presentation_id=pid, call=call, answer_attempt=answered + 1)
            try:
                res = TRANSPORT(args.model, args.agent, prompt, shown, args.timeout)
            except Exception as exc:  # noqa: BLE001  (no output observed: a transport failure)
                res = {"response": "", "tool_calls": [], "blinding": "not_checked", "returncode": "exception",
                       "stderr_tail": f"{type(exc).__name__}: {exc}", "seconds": None}
            entry.update(res)
            if res["returncode"] != 0 or res["blinding"] != "ok" or not (res["response"] or "").strip():
                entry["parse_status"] = (f"transport_rc_{res['returncode']}" if res["returncode"] != 0
                                         else f"blinding_{res['blinding']}" if res["blinding"] != "ok"
                                         else "empty_response")
                journal.log(type="end", presentation_id=pid, call=call, entry=entry)
                time.sleep(min(60, 10 * call) if args.backoff else 0)
                continue
            answered += 1
            entry["counted_answer"] = True
            answer, why = parse_answer(res["response"], require_think_end=False)
            entry["parse_status"] = why
            journal.log(type="end", presentation_id=pid, call=call, entry=entry)
            if answer is not None:
                rec.update({"status": "ok", "parsed": answer})
                break
            if answered == 2:
                rec["status"] = "parse_failed"
                break
        else:
            rec["status"] = "error"
            rec["error"] = f"no terminal outcome within {args.max_attempts} calls"
    except Exception as exc:  # noqa: BLE001  (recorded, never dropped)
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    return rec


def recover(out, jpath, manifest_hash, plan, stim_hash, accept_interrupted):
    """Read records + journal. Returns (done, prior answers, interrupted counts, records to write).

    Per presentation without a terminal record: answers spent = counted 'end' entries in the journal
    (all invocations); a terminal outcome present in the journal but never written (an 'ok' answer, or
    two unparseable answers) is recovered as a record without a new call; a call with a 'start' but no
    'end' (interrupted, output never observed) blocks the resume unless accept_interrupted."""
    done, err_recs = set(), {}
    for line in out.read_text().splitlines() if out.exists() else []:
        r = json.loads(line)
        if r.get("manifest_sha256") != manifest_hash:
            raise SystemExit(f"record {r.get('presentation_id')} belongs to a different manifest")
        pid = r["presentation_id"]
        if pid in done:
            raise SystemExit(f"{pid}: record after a terminal record")
        if r.get("status") in ("ok", "parse_failed"):
            if r.get("image_sha256") != [stim_hash[r["pair_id"]][k] for k in shown_keys(r)]:
                raise SystemExit(f"completed record {pid} used different stimuli")
            done.add(pid)
        elif r.get("status") == "error":
            err_recs.setdefault(pid, []).append(r)
        else:
            raise SystemExit(f"{pid}: unknown status {r.get('status')!r}")
    starts, ends = {}, {}
    for line in jpath.read_text().splitlines() if jpath.exists() else []:
        ev = json.loads(line)
        if ev.get("manifest_sha256") != manifest_hash:
            raise SystemExit("journal entry belongs to a different manifest")
        if ev["type"] == "accept_interrupted":
            continue                       # disclosure event; the counts below come from start/end pairs
        if ev["type"] not in ("start", "end"):
            raise SystemExit(f"unknown journal event {ev['type']!r}")
        key = (ev["invocation"], ev["call"])
        (starts if ev["type"] == "start" else ends).setdefault(ev["presentation_id"], {})[key] = ev
    prior, interrupted, recovered, to_accept = {}, {}, [], {}
    step_of = {s["presentation_id"]: s for s in plan}
    for pid in sorted(set(starts) | set(ends) | set(err_recs)):
        if pid in done:
            continue
        e_pid = ends.get(pid, {})
        counted = sum(bool(ev["entry"]["counted_answer"]) for ev in e_pid.values())
        rec_invs = {r["invocation"] for r in err_recs.get(pid, [])}
        in_records = sum(bool(a.get("counted_answer")) for r in err_recs.get(pid, []) for a in r["attempts"])
        if in_records != sum(bool(ev["entry"]["counted_answer"]) for (inv, _), ev in e_pid.items() if inv in rec_invs):
            raise SystemExit(f"{pid}: error records and journal disagree on answers spent")
        # calls started but never finished, in ANY earlier invocation (journal-derived, so the disclosure
        # survives any number of interruptions and recoveries)
        lost = sorted(set(starts.get(pid, {})) - set(e_pid))
        oks = [ev["entry"] for ev in e_pid.values() if ev["entry"]["parse_status"] == "ok"]
        if len(oks) > 1:
            raise SystemExit(f"{pid}: more than one parsed answer in the journal")
        if oks or counted >= 2:
            s = step_of[pid]
            answer = parse_answer(oks[0]["response"], require_think_end=False)[0] if oks else None
            rec = {"presentation_id": pid, "pair_id": s["pair_id"], "condition": s["condition"], "order": s["order"],
                   "manifest_sha256": manifest_hash, "recovered_from_journal": True,
                   "interrupted_calls_before": len(lost),
                   "attempts": [ev["entry"] for (inv, _), ev in sorted(e_pid.items(), key=lambda kv: kv[1]["t"])
                                if inv not in rec_invs],
                   "image_sha256": [stim_hash[s["pair_id"]][k] for k in shown_keys(s)],
                   "status": "ok" if answer else "parse_failed", **({"parsed": answer} if answer else {})}
            recovered.append(rec)
            continue
        if lost and not accept_interrupted:
            raise SystemExit(f"{pid}: {len(lost)} interrupted call(s) whose output was never observed; "
                             f"resume with --accept_interrupted to treat them as unanswered transport failures")
        prior[pid], interrupted[pid] = counted, len(lost)
        if lost:
            to_accept[pid] = [list(k) for k in lost]
    return done, prior, interrupted, recovered, to_accept


def main(argv=None):
    import numpy as np
    from render_cards import PROTO_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--agent", default="blind-reader", choices=["blind-reader"])   # its definition is hashed
    ap.add_argument("--agent_definition_dir", default=str(AGENT_DIR),
                    help="directory of the installed blind-reader agent definition (plugin.json, agents/)")
    ap.add_argument("--agent_cli", default=None, help="agentic CLI executable (default: $GEMINI_AGENT_CLI or agy)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max_attempts", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only (first N planned presentations)")
    ap.add_argument("--no_backoff", dest="backoff", action="store_false")
    ap.add_argument("--accept_interrupted", action="store_true",
                    help="resume past calls that were started but never finished (output never observed), "
                         "counting them as unanswered transport failures (recorded per presentation)")
    args = ap.parse_args(argv)
    if args.agent_cli:
        gj.set_agent_cli(args.agent_cli)

    items_bytes = Path(args.items).read_bytes()
    doc = json.loads(items_bytes)
    by_id = {p["pair_id"]: p for p in doc["pairs"]}
    full_plan = doc["plan"]
    plan = list(full_plan)                       # one shard: the whole plan, in plan order
    run_plan = plan[: args.limit] if args.limit else plan
    bb = np.load(PROTO_DIR / "bb4.npy")
    needed = sorted({s["pair_id"] for s in plan})
    stimuli = {pid: render_stimuli(by_id[pid], doc["class_names"], bb) for pid in needed}
    stim_hash = {pid: {k: sha256_bytes(v) for k, v in d.items()} for pid, d in stimuli.items()}
    agent_sha = {f: sha256_bytes((Path(args.agent_definition_dir) / f).read_bytes()) for f in AGENT_FILES}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "items_sha256": sha256_bytes(items_bytes), "items_kind": doc.get("kind"),
        "full_plan_sha256": sha256_bytes(json.dumps(full_plan, sort_keys=True).encode()),
        "num_shards": 1, "shard_index": 0,
        "plan_sha256": sha256_bytes(json.dumps(plan, sort_keys=True).encode()),
        "stimuli_sha256": sha256_bytes(json.dumps(stim_hash, sort_keys=True).encode()),
        "prompt_sha256": sha256_bytes((PREAMBLE + PROMPT + RETRY_SUFFIX).encode()),
        "model_id": f"agy:{args.model}", "agent": args.agent, "agent_definition_sha256": agent_sha,
        "transport": "antigravity-cli stream-json, neutral temp dir, verified read_file-only blinding",
        "enable_thinking": "provider (model name)", "sampling": "provider default (no seed)", "seed": None,
        "require_think_end": False, "max_attempts": args.max_attempts, "timeout": args.timeout,
        "code_sha256": {f: sha256_bytes((HERE / f).read_bytes())
                        for f in ("narrator_gemini_run.py", "narrator_run.py", "narrator_board.py",
                                  "render_cards.py", "gemini_judge.py", "vlm_judge.py")},
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
    manifest_hash = sha256_bytes(json.dumps(manifest, sort_keys=True).encode())
    jpath = out.with_suffix(".journal.jsonl")
    if out.exists() and out.stat().st_size > 0 and not jpath.exists():
        raise SystemExit(f"{out} has records but no journal; refusing to resume")
    done, prior, interrupted, recovered, to_accept = recover(out, jpath, manifest_hash, plan, stim_hash,
                                                             args.accept_interrupted)
    for rec in recovered:          # terminal outcomes that were reached but never written
        pr = by_id[rec["pair_id"]]
        rec.update({"corruption": pr["corruption"], "stratum": pr["stratum"]})
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        done.add(rec["presentation_id"])
        print(f"[gemini-narrator] {rec['presentation_id']} -> {rec['status']} (recovered from journal)", flush=True)
    todo = [s for s in run_plan if s["presentation_id"] not in done]
    print(f"[gemini-narrator] {len(done)} done, {len(todo)} to run, model={args.model} agent={args.agent}", flush=True)
    journal = Journal(jpath, manifest_hash, uuid.uuid4().hex)
    for pid, lost in sorted(to_accept.items()):   # persisted before any new call of this invocation
        journal.log(type="accept_interrupted", presentation_id=pid, interrupted_calls=lost)
    lock = threading.Lock()
    with out.open("a") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_presentation, s, by_id[s["pair_id"]], stimuli[s["pair_id"]], manifest_hash, args,
                          journal, prior.get(s["presentation_id"], 0), interrupted.get(s["presentation_id"], 0)): s
                for s in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            p = rec.get("parsed", {})
            secs = sum(a.get("seconds") or 0 for a in rec["attempts"])
            print(f"[gemini-narrator] {rec['presentation_id']} -> {rec['status']} calls={len(rec['attempts'])} "
                  f"{secs:.0f}s {({k: p.get(k) for k in FIELDS}) if p else rec.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
