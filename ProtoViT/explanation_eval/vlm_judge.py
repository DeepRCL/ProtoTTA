#!/usr/bin/env python
"""Blinded pairwise VLM judge (Qwen3.6-35B-A3B) on the same cards as the human study.

For each item the judge receives the damaged test photo and the explanation cards of ProtoTTA and
EATA (labelled A and B only; no class names, scores or method names) together with the
instructions humans get, and answers "Which explanation is better?" (A / B / SAME). Images are
passed as in-memory pixels (no file names reach the model). Cards are rendered with
render_cards.item_images (the code that produced the human-study release; boxes drawn on the clean
copy of the photo); the sha256 of every image shown is stored with the record.
Position bias: --schedule both_orders shows every item in both orders (each method once at A and
once at B); which order comes first is a seeded coin flip per item.
Auditing: every attempt (prompt, raw output, new tokens, token-limit flag, time,
status/exception) is stored; render/inference exceptions are recorded, never
dropped. A run manifest (items sha256, schedule/plan, prompt, model, generation
settings and a digest of every stimulus image) is written on first start and must
match exactly on resume; completed records must carry the same image hashes.
"""
import argparse
import hashlib
import io
import json
import random
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PV = HERE.parent

LEFT_PANEL = {
    # human-study format (boxes drawn on an undamaged copy)
    "clean": ("on the left is an undamaged copy of the same photo, with small coloured boxes marking the regions "
              "the AI relied on when it looked at the damaged photo (the boxes are drawn on the undamaged copy so "
              "you can see which parts of the bird they cover)"),
}

# Pairwise design (ProtoTTA vs EATA): one question, the same wording humans get.
PAIR_METHODS = ("eata", "prototta")
PAIR_INSTRUCTIONS = """You are helping evaluate explanations produced by an AI system that recognizes bird species in photos damaged by noise, blur, fog, compression and similar effects.

IMAGE 1 is the damaged photo the AI saw.
IMAGES 2 and 3 are two different explanations of the AI's decision, called Explanation A and Explanation B (in that order).

How to read an explanation: {LEFT_PANEL}; boxes of one colour (and number) belong together. On the right are five training examples; example k has a frame of the same colour as group k on the left, and its own boxes show the region the AI thinks looks alike.
A good explanation puts its boxes on meaningful parts of the bird (head, beak, eye, wing, tail, body markings) and matches them to similar parts in the training examples. A poor one points at background, noise or damage, or matches regions that do not look alike. You do not need to know the bird species.

Question: Which explanation is better?
Answer "A" or "B"; answer "SAME" only if you see no clear difference between them.

Return exactly one JSON object and nothing else:
{"better": "A|B|SAME", "rationale": "<one short sentence>"}"""
DESIGNS = {
    "pairwise": {"methods": PAIR_METHODS, "labels": ("A", "B"), "choices": {"A", "B", "SAME"}, "questions": ("better",),
                 "template": PAIR_INSTRUCTIONS, "no_choice": "SAME", "schedules": ("both_orders",)},
}


def instructions_for(background, design="pairwise"):
    return DESIGNS[design]["template"].replace("{LEFT_PANEL}", LEFT_PANEL[background])


def choice_methods(answer, order, design="pairwise"):
    """Map position letters to methods; 'same' (SAME) otherwise."""
    d = DESIGNS[design]
    pos = dict(zip(d["labels"], order))
    other = "same"
    return {q: pos.get(answer[q], other) for q in d["questions"]}


RETRY_SUFFIX = "\nReturn ONLY the JSON object."
THINK_END = "</think>"
# Qwen's documented thinking-budget recipe (Qwen3 model card, "thinking budget"): when the
# reasoning exceeds the budget, append this text (which closes the thinking block) and let
# the model write its final answer. Such records are flagged forced_think_end=True.
EARLY_STOP_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution "
                   "based on the thinking directly now.\n</think>.\n\n")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def parse_answer(text, require_think_end, design="pairwise"):
    """Parse the final answer. If the model reasons (thinking mode), only text after
    the closing </think> is eligible; unfinished reasoning is rejected."""
    d = DESIGNS[design]
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
        vals = {k: str(obj.get(k, "")).strip().upper() for k in d["questions"]}
        if all(v in d["choices"] for v in vals.values()):
            vals["rationale"] = str(obj.get("rationale", ""))[:500]
            found = vals  # keep the last valid object
    return (found, "ok") if found else (None, "no_valid_json")


class Judge:
    def __init__(self, model_id, max_new_tokens, enable_thinking=None, think_budget=None, answer_budget=1024,
                 sampling="greedy", early_stop_text=None):
        from transformers import AutoProcessor
        kw = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": "auto"}
        if "Qwen3.6" in model_id:
            from transformers import AutoModelForMultimodalLM
            self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw)
        else:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kw)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.think_budget = think_budget
        self.answer_budget = answer_budget
        # "greedy": do_sample=False. "recommended": the model's own generation_config sampling
        # parameters (Qwen advises against greedy decoding in thinking mode), seeded per call.
        assert sampling in ("greedy", "recommended"), sampling
        self.sampling = sampling
        # text appended when the thinking budget is exhausted (default: Qwen's recipe, unchanged for
        # existing runs; the narrator appends the start of its JSON answer to it)
        self.early_stop_text = EARLY_STOP_TEXT if early_stop_text is None else early_stop_text

    def _gen(self, inputs, max_new_tokens, seed):
        if self.sampling == "greedy":
            return self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

    def _decode(self, ids):
        # keep special/added tokens so the </think> boundary is visible to the parser
        text = self.processor.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text

    @torch.no_grad()
    def __call__(self, pil_images, prompt, seed=0):
        content = [{"type": "image", "image": im} for im in pil_images] + [{"type": "text", "text": prompt}]
        kw = {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **kw).to(self.model.device)
        n_prompt = inputs["input_ids"].shape[1]
        # Uniform per-attempt metadata. first_stage_tokens = tokens of the first generate() call
        # (reasoning and, if it closed, possibly the answer); continuation_tokens = tokens generated
        # after the thinking budget was exhausted (0 if not needed).
        if self.think_budget is None:
            out = self._gen(inputs, self.max_new_tokens, seed)
            gen = out[0][n_prompt:]
            n1 = int(gen.shape[0])
            return self._decode(gen), n1, {"forced_think_end": False, "continued_after_budget": False,
                                          "first_stage_tokens": n1, "continuation_tokens": 0,
                                          "hit_token_limit": n1 >= self.max_new_tokens and not self._ended(gen)}
        out = self._gen(inputs, self.think_budget, seed)
        gen = out[0][n_prompt:]
        n1 = int(gen.shape[0])
        text = self._decode(gen)
        if n1 < self.think_budget or self._ended(gen):
            # generation finished within the budget (EOS): identical to an unbudgeted run
            return text, n1, {"forced_think_end": False, "continued_after_budget": False,
                              "first_stage_tokens": n1, "continuation_tokens": 0, "hit_token_limit": False}
        # budget exhausted without EOS. If the reasoning already closed, continue the answer (same
        # as an unbudgeted run with a larger limit); otherwise close it with EARLY_STOP_TEXT.
        closed = THINK_END in text
        stop_ids = out[0].new_empty(0) if closed else self.processor.tokenizer(
            self.early_stop_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(out.device)
        ids = torch.cat([out[0], stop_ids]).unsqueeze(0)
        n_ext = ids.shape[1] - n_prompt
        inputs2 = {}
        for k, v in inputs.items():
            if k == "input_ids":
                inputs2[k] = ids
            elif k == "attention_mask":
                inputs2[k] = torch.ones_like(ids)
            elif torch.is_tensor(v) and v.dim() == 2 and v.shape == inputs["input_ids"].shape:
                # per-token side inputs (e.g. multimodal token types): appended tokens are text (0)
                inputs2[k] = torch.cat([v, v.new_zeros((1, n_ext))], dim=1)
            else:
                inputs2[k] = v
        out2 = self._gen(inputs2, self.answer_budget, seed + 1)
        gen2 = out2[0][ids.shape[1]:]
        n2 = int(gen2.shape[0])
        text_all = text + ("" if closed else self._decode(stop_ids)) + self._decode(gen2)
        return text_all, n1 + n2, {"forced_think_end": not closed, "continued_after_budget": closed,
                                   "first_stage_tokens": n1, "continuation_tokens": n2,
                                   "hit_token_limit": n2 >= self.answer_budget and not self._ended(gen2)}

    def _ended(self, gen):
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        return gen.shape[0] > 0 and int(gen[-1]) in eos


def render_item_images(item, bb, background="clean", methods=PAIR_METHODS):
    from render_cards import item_images  # single source of truth, same as the website release
    imgs = item_images(item["dataset_root"], item["path"], {m: item["methods"][m]["top"] for m in methods}, bb,
                       background=background)
    out = {}
    for k, im in imgs.items():
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92 if k == "test" else 90)
        out[k] = buf.getvalue()
    return out


def build_schedule(items, mode, seed, methods=PAIR_METHODS):
    rng = random.Random(seed)
    plan = []
    assert mode == "both_orders", mode
    # every item in both orders (each method once at A and once at B);
    # which order comes first (presentation 0) is a seeded coin flip per item
    assert len(methods) == 2, methods
    for it in items:
        base = list(methods)
        rng.shuffle(base)
        plan.append({"item_id": it["item_id"], "presentation": 0, "order": base})
        plan.append({"item_id": it["item_id"], "presentation": 1, "order": base[::-1]})
    return plan


def position_counts(plan, labels=("A", "B")):
    counts = {}
    for step in plan:
        for pos, m in zip(labels, step["order"]):
            counts[(m, pos)] = counts.get((m, pos), 0) + 1
    return counts


def main():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(HERE))
    from render_cards import PROTO_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--card_background", choices=["clean"], default="clean",
                    help="human-study format: boxes drawn on the undamaged copy of the photo")
    ap.add_argument("--model_id", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--think_budget", type=int, default=None,
                    help="thinking-budget mode: reasoning capped at this many tokens, then EARLY_STOP_TEXT "
                         "and up to --answer_budget answer tokens (replaces --max_new_tokens)")
    ap.add_argument("--answer_budget", type=int, default=1024)
    ap.add_argument("--enable_thinking", choices=["default", "on", "off"], default="default")
    ap.add_argument("--design", choices=sorted(DESIGNS), default="pairwise")
    ap.add_argument("--schedule", choices=["both_orders"], default="both_orders")
    ap.add_argument("--sampling", choices=["greedy", "recommended"], default="greedy")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only (first N planned presentations)")
    ap.add_argument("--num_shards", type=int, default=1, help="split the (deterministic) plan across jobs")
    ap.add_argument("--shard_index", type=int, default=0)
    args = ap.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    design = DESIGNS[args.design]
    if args.schedule not in design["schedules"]:
        raise SystemExit(f"schedule {args.schedule} is not valid for design {args.design}")
    methods, labels = design["methods"], design["labels"]
    instr = instructions_for(args.card_background, args.design)

    items_bytes = Path(args.items).read_bytes()
    items_doc = json.loads(items_bytes)
    items = items_doc["items"]
    if items_doc.get("design") != args.design:
        raise SystemExit(f"items file is design {items_doc.get('design')!r}, not {args.design!r}")
    by_id = {it["item_id"]: it for it in items}
    full_plan = build_schedule(items, args.schedule, args.seed, methods)
    full_plan_sha = sha256_bytes(json.dumps(full_plan, sort_keys=True).encode())
    plan = [s for k, s in enumerate(full_plan) if k % args.num_shards == args.shard_index]
    if args.limit:
        plan = plan[: args.limit]
    release = None
    bb = np.load(PROTO_DIR / "bb4.npy")
    needed = sorted({s["item_id"] for s in plan})
    stimuli, prep_failures = {}, []
    for i in needed:   # all rendered up front; failures are persisted, then the run stops
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
    # Qwen3-VL-*-Thinking and Qwen3.6 (thinking by default) must close their reasoning
    require_think_end = args.enable_thinking != "off" and ("Thinking" in args.model_id or "Qwen3.6" in args.model_id)
    print("[vlm] method-position counts:", position_counts(plan, labels), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "items_sha256": sha256_bytes(items_bytes), "release": release, "schedule": args.schedule,
        "seed": args.seed, "full_plan_sha256": full_plan_sha, "num_shards": args.num_shards,
        "shard_index": args.shard_index, "plan_sha256": sha256_bytes(json.dumps(plan, sort_keys=True).encode()),
        "stimuli_sha256": sha256_bytes(json.dumps(stim_hash, sort_keys=True).encode()),
        "card_background": args.card_background,
        "prompt_sha256": sha256_bytes((instr + RETRY_SUFFIX).encode()), "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "require_think_end": require_think_end,
        "design": args.design,
        **({"sampling": args.sampling} if args.sampling != "greedy" else {}),
        **({"think_budget": args.think_budget, "answer_budget": args.answer_budget,
            "early_stop_text": EARLY_STOP_TEXT} if args.think_budget else {}),
        "render_code_sha256": sha256_bytes((HERE / "render_cards.py").read_bytes()),
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
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                expect = [stim_hash[r["item_id"]]["test"]] + [stim_hash[r["item_id"]][m] for m in r["order"]]
                if r.get("image_sha256") != expect:
                    raise SystemExit(f"completed record {r['item_id']} p{r['presentation']} used different stimuli")
                done.add((r["item_id"], r["presentation"]))
    think = {"default": None, "on": True, "off": False}[args.enable_thinking]
    judge = Judge(args.model_id, args.max_new_tokens, think, args.think_budget, args.answer_budget, args.sampling)
    with out.open("a") as fh:
        for step in plan:
            key = (step["item_id"], step["presentation"])
            if key in done:
                continue
            it = by_id[step["item_id"]]
            rec = {"item_id": it["item_id"], "corruption": it["corruption"], "sample_index": it["sample_index"],
                   "presentation": step["presentation"], "order": step["order"], "manifest": manifest,
                   "attempts": [], "status": "pending"}
            try:
                imgs = stimuli[it["item_id"]]
                shown = [imgs["test"]] + [imgs[m] for m in step["order"]]
                rec["image_sha256"] = [sha256_bytes(b) for b in shown]
                pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in shown]
                for attempt in (1, 2):
                    prompt = instr if attempt == 1 else instr + RETRY_SUFFIX
                    entry = {"attempt": attempt, "prompt": prompt, "raw": None, "new_tokens": None,
                             "hit_token_limit": None, "seconds": None, "parse_status": "inference_error"}
                    rec["attempts"].append(entry)        # recorded before inference
                    t0 = time.time()
                    try:
                        call_seed = int(sha256_bytes(f"{args.seed}:{it['item_id']}:{step['presentation']}:{attempt}"
                                                     .encode())[:8], 16)
                        raw, n_new, gen_info = judge(pil, prompt, call_seed)
                        entry["seed"] = call_seed
                    except Exception as exc:
                        entry.update({"seconds": round(time.time() - t0, 1),
                                      "error": f"{type(exc).__name__}: {exc}"})
                        raise
                    answer, why = parse_answer(raw, require_think_end, args.design)
                    entry.update({"raw": raw, "new_tokens": n_new, **gen_info,
                                  "seconds": round(time.time() - t0, 1), "parse_status": why})
                    if answer is not None:
                        rec.update({"status": "ok", "parsed": answer,
                                    "choice_methods": choice_methods(answer, step["order"], args.design)})
                        break
                else:
                    rec["status"] = "parse_failed"
            except Exception as exc:  # recorded, never dropped
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()[-4000:]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[vlm] {rec['item_id']} p{rec['presentation']} {rec['order']} -> {rec['status']} "
                  f"{rec.get('choice_methods')}", flush=True)


if __name__ == "__main__":
    main()
