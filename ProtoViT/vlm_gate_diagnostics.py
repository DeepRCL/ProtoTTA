#!/usr/bin/env python3
"""Focused diagnostics for the full-set VLM ProtoTTA gate.

This experiment reuses the completed, label-blind evidence export.  Ground
truth is used only to construct a balanced diagnostic manifest and to evaluate
the saved VLM choices; it is never rendered or included in a prompt.

The matched conditions isolate likely failures in the full-set experiment:

* image_nomsp: raw image and candidate labels only;
* image_msp: add the two MSP values;
* compact_msp: the compact board used by the full-set experiment;
* compact_nomsp: remove confidence from that board/prompt;
* rich_nomsp: all five predicted-class and ten any-class prototypes per state;
* rich_nomsp_reverse: identical rich evidence with the spatial state order
  reversed, measuring position sensitivity.
* candidate_gallery_nomsp: a candidate-aligned comparison that presents both
  predicted species' prototype galleries symmetrically rather than attaching
  self-confirming prototypes to their originating state.

Every call contains one bird only.  A/B next-token probabilities avoid JSON
generation failures and provide a continuous label-blind preference score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import fullset_vlm_gate as full
from paired_adaptation_supervision import choice_probability


FAST_CONDITIONS = [
    "image_nomsp",
    "image_msp",
    "compact_msp",
    "compact_nomsp",
    "rich_nomsp",
    "rich_nomsp_reverse",
    "candidate_gallery_nomsp",
]
REASONED_CONDITIONS = [
    "reasoned_image_nomsp",
    "reasoned_rich_nomsp",
    "skeptical_rich_nomsp",
]
CONDITIONS = FAST_CONDITIONS + REASONED_CONDITIONS
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
LOGGER = logging.getLogger("vlm_gate_diagnostics")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullset-dir", type=Path, default=root / "results" / "vlm_fullset_gate")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "vlm_gate_diagnostics")
    parser.add_argument("--data-dir", type=Path, default=root / "datasets" / "cub200_c")
    parser.add_argument(
        "--prototype-dir",
        type=Path,
        default=root / "saved_models" / "deit_small_patch16_224" / "exp1" / "img" / "epoch-4",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--per-outcome-per-corruption", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=20260811)
    score = sub.add_parser("score")
    score.add_argument("--condition", choices=CONDITIONS, required=True)
    score.add_argument("--model-id", default=MODEL_ID)
    score.add_argument("--max-new-tokens", type=int, default=1536)
    score.add_argument("--max-samples", type=int)
    score.add_argument("--overwrite", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--bootstrap-resamples", type=int, default=5000)
    analyze.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def deterministic_order(corruption: str, key: str, reverse: bool = False) -> Tuple[str, str]:
    digest = hashlib.sha256(f"vlm-gate-diagnostic::{corruption}::{key}".encode()).digest()
    order = ("before", "after") if digest[0] % 2 == 0 else ("after", "before")
    return order[::-1] if reverse else order


def run_prepare(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    samples = []
    for corruption in full.CORRUPTIONS:
        export = read_json(full.export_path(args.fullset_dir, corruption))
        groups = {"beneficial": [], "harmful": []}
        for key in export["evidence"]:
            record = export["records"][key]
            if record["after_correct"] and not record["before_correct"]:
                groups["beneficial"].append(key)
            elif record["before_correct"] and not record["after_correct"]:
                groups["harmful"].append(key)
        for outcome, keys in groups.items():
            ordered = sorted(keys)
            rng.shuffle(ordered)
            selected = ordered[: args.per_outcome_per_corruption]
            if len(selected) != args.per_outcome_per_corruption:
                raise RuntimeError(f"Insufficient {outcome} cases for {corruption}: {len(selected)}")
            for key in selected:
                samples.append(
                    {
                        "corruption": corruption,
                        "key": key,
                        "public_id": full.public_id(key),
                        "outcome": outcome,
                    }
                )
    payload = {
        "schema_version": 1,
        "label_visible_to_vlm": False,
        "balanced_by_outcome_and_corruption": True,
        "per_outcome_per_corruption": args.per_outcome_per_corruption,
        "seed": args.seed,
        "num_samples": len(samples),
        "samples": samples,
    }
    write_json(args.output_dir / "manifest.json", payload)
    LOGGER.info("Wrote balanced manifest with %d samples", len(samples))


def _state_title(prefix: str) -> str:
    return "BEFORE: FROZEN SOURCE" if prefix == "before" else "AFTER: CONTINUOUS PROTOTTA"


def _state_color(prefix: str) -> str:
    return "#285A8E" if prefix == "before" else "#A54822"


def _prototype_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    prototypes: Sequence[Dict],
    prototype_dir: Path,
    y: int,
    columns: int,
    title: str,
) -> int:
    draw.text((30, y), title, fill="#202020", font=full.font(25, True))
    y += 42
    card_w, image_h, gap_x, gap_y = 250, 145, 26, 58
    for index, proto in enumerate(prototypes):
        row, column = divmod(index, columns)
        x = 30 + column * (card_w + gap_x)
        top = y + row * (image_h + gap_y)
        patch = full.prototype_image(prototype_dir, proto, (card_w, image_h))
        canvas.paste(patch, (x, top))
        label = str(proto.get("class_name", "prototype"))[:27]
        detail = f"{label} | P{proto['proto_idx']} | {float(proto.get('contribution', 0)):.2f}"
        draw.text((x, top + image_h + 5), detail, fill="#202020", font=full.font(15))
    rows = (len(prototypes) + columns - 1) // columns
    return y + rows * (image_h + gap_y)


def rich_state_board(
    raw_path: Path,
    record: Dict,
    evidence: Dict,
    prefix: str,
    prototype_dir: Path,
    destination: Path,
) -> None:
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGB")
    canvas = Image.new("RGB", (1436, 1310), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1436, 105), fill=_state_color(prefix))
    draw.text((28, 14), _state_title(prefix), fill="white", font=full.font(31, True))
    draw.text(
        (28, 58),
        f"Prediction: {record[prefix + '_prediction']}",
        fill="white",
        font=full.font(23, True),
    )
    canvas.paste(ImageOps.fit(raw, (330, 330), method=Image.Resampling.LANCZOS), (80, 130))
    canvas.paste(full.focus_overlay(raw, evidence[prefix + "_predicted"], 330), (520, 130))
    draw.text((80, 468), "Corrupted input", fill="#202020", font=full.font(21, True))
    draw.text((520, 468), "Prototype focus", fill="#202020", font=full.font(21, True))
    y = _prototype_grid(
        canvas,
        draw,
        evidence[prefix + "_predicted"][:5],
        prototype_dir,
        520,
        5,
        "All five predicted-class prototypes",
    )
    _prototype_grid(
        canvas,
        draw,
        evidence[prefix + "_any"][:10],
        prototype_dir,
        y + 6,
        5,
        "Ten strongest any-class / competing prototypes",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, optimize=True)


def compact_ordered_canvas(
    raw_path: Path,
    record: Dict,
    evidence: Dict,
    order: Tuple[str, str],
    prototype_dir: Path,
    destination: Path,
    show_msp: bool,
) -> None:
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGB")
    side_width, height, gutter = 860, 1030, 24
    canvas = Image.new("RGB", (2 * side_width + gutter, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, prefix in enumerate(order):
        x0 = column * (side_width + gutter)
        draw.rectangle((x0, 0, x0 + side_width, 120), fill=_state_color(prefix))
        draw.text((x0 + 20, 15), _state_title(prefix), fill="white", font=full.font(30, True))
        subtitle = f"Prediction: {record[prefix + '_prediction']}"
        if show_msp:
            subtitle += f" | MSP {record[prefix + '_msp']:.3f}"
        draw.text((x0 + 20, 61), subtitle, fill="white", font=full.font(23, True))
        predicted = evidence[prefix + "_predicted"]
        any_class = evidence[prefix + "_any"]
        competitors = [
            item for item in any_class
            if int(item["class_index"]) != int(record[prefix + "_prediction_index"])
        ] or list(any_class)
        canvas.paste(ImageOps.fit(raw, (300, 300), method=Image.Resampling.LANCZOS), (x0 + 60, 145))
        canvas.paste(full.focus_overlay(raw, predicted, 300), (x0 + 500, 145))
        draw.text((x0 + 60, 452), "Corrupted input", fill="#222", font=full.font(20, True))
        draw.text((x0 + 500, 452), "Prototype focus", fill="#222", font=full.font(20, True))
        full._draw_prototype_row(canvas, draw, x0, 495, side_width, "Top predicted-class prototypes", predicted, prototype_dir)
        full._draw_prototype_row(canvas, draw, x0, 755, side_width, "Strong competing-class prototypes", competitors, prototype_dir)
    draw.rectangle((side_width, 0, side_width + gutter, height), fill="#222")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90, optimize=True)


def candidate_gallery_canvas(
    raw_path: Path,
    record: Dict,
    evidence: Dict,
    order: Tuple[str, str],
    prototype_dir: Path,
    destination: Path,
) -> None:
    """Render both candidate classes on one common, state-neutral basis."""
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGB")
    canvas = Image.new("RGB", (1500, 1240), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1500, 105), fill="#343A40")
    draw.text((30, 18), "CANDIDATE-ALIGNED PROTOTYPE COMPARISON", fill="white", font=full.font(31, True))
    canvas.paste(ImageOps.fit(raw, (360, 360), method=Image.Resampling.LANCZOS), (30, 135))
    draw.text((30, 505), "Corrupted input", fill="#202020", font=full.font(22, True))
    for index, (label, prefix) in enumerate(zip(("A", "B"), order)):
        x = 435 + index * 520
        canvas.paste(full.focus_overlay(raw, evidence[prefix + "_predicted"], 360), (x, 135))
        role = "BEFORE" if prefix == "before" else "AFTER"
        draw.text((x, 505), f"STATE {label} ({role}) focus", fill="#202020", font=full.font(21, True))
    candidates = [
        ("A", order[0], evidence[order[0] + "_predicted"][:5]),
        ("B", order[1], evidence[order[1] + "_predicted"][:5]),
    ]
    for row, (label, prefix, prototypes) in enumerate(candidates):
        y = 560 + row * 330
        draw.rectangle((0, y, 1500, y + 48), fill="#E9EEF3")
        draw.text(
            (30, y + 9),
            f"CANDIDATE {label}: {record[prefix + '_prediction']} — five reference prototypes",
            fill="#202020",
            font=full.font(24, True),
        )
        for column, proto in enumerate(prototypes):
            x = 30 + column * 288
            patch = full.prototype_image(prototype_dir, proto, (260, 180))
            canvas.paste(patch, (x, y + 62))
            detail = f"P{proto['proto_idx']} | evidence {float(proto.get('contribution', 0)):.2f}"
            draw.text((x, y + 248), detail, fill="#202020", font=full.font(16))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, optimize=True)


def build_prompt(record: Dict, order: Tuple[str, str], condition: str) -> str:
    state_lines = []
    for label, prefix in zip(("A", "B"), order):
        role = "BEFORE frozen source" if prefix == "before" else "AFTER continuous ProtoTTA"
        detail = f"STATE {label} is {role} and predicts {record[prefix + '_prediction']}"
        if condition in {"image_msp", "compact_msp"}:
            detail += f" (MSP {record[prefix + '_msp']:.3f})"
        state_lines.append(detail + ".")
    if condition.startswith("image_"):
        evidence = "IMAGE 1 is the corrupted bird image; no prototype board is supplied."
    elif condition.startswith("compact_"):
        evidence = (
            "IMAGE 1 is one compact paired board whose left and right columns are explicitly labelled. "
            "Each state shows the corrupted input, focus, three predicted-class prototypes, and three competitors."
        )
    elif condition == "candidate_gallery_nomsp":
        evidence = (
            "IMAGE 1 is a candidate-aligned comparison: the raw bird, STATE A and B focus maps, and two symmetric "
            "five-prototype reference galleries for the two predicted species. The galleries are candidates, not proof "
            "that their originating state is correct."
        )
    else:
        evidence = (
            "IMAGE 1 is the corrupted bird. IMAGE 2 is STATE A's rich reasoning board and IMAGE 3 is STATE B's. "
            "Each rich board contains all five predicted-class prototypes and ten any-class competitors."
        )
    return (
        "You are a label-blind supervisor of a prototype bird classifier. Ground truth and correctness are withheld. "
        "Adaptation may help or harm; do not prefer AFTER merely because it is newer.\n\n"
        + "\n".join(state_lines)
        + "\n\n"
        + evidence
        + "\nCompare visible species fit, localization on real bird anatomy, prototype consistency, and competing evidence. "
        "Choose the state whose prediction is more likely correct. Your entire answer must be one character: A or B.\nANSWER:"
    )


def build_reasoned_prompt(record: Dict, order: Tuple[str, str], condition: str) -> str:
    state_lines = []
    for label, prefix in zip(("A", "B"), order):
        role = "BEFORE frozen source" if prefix == "before" else "AFTER continuous ProtoTTA"
        state_lines.append(
            f"STATE {label} is {role} and predicts {record[prefix + '_prediction']}."
        )
    if condition == "reasoned_image_nomsp":
        evidence = (
            "IMAGE 1 is the corrupted bird image. No classifier reasoning board is supplied, so prototype "
            "or localization changes are not observable. Judge the two species labels against visible anatomy."
        )
    else:
        evidence = (
            "IMAGE 1 is the corrupted bird. IMAGE 2 is STATE A's rich reasoning board and IMAGE 3 is STATE B's. "
            "Each state board contains the classifier's focus map, all five predicted-class prototypes, and ten "
            "strong any-class/competing prototypes. Prototype patches are stored training exemplars, not crops "
            "from this test image. A state's own retrieved prototypes are hypotheses, not proof it is correct."
        )
    return (
        "You are an independent, label-blind supervisor of a prototype-based bird classifier. Ground truth and "
        "correctness are withheld. Decide whether the AFTER adaptation is genuinely helpful for this sample. "
        "Do not prefer AFTER because it is newer, more confident, or visually placed in a particular position.\n\n"
        + "\n".join(state_lines)
        + "\n\n"
        + evidence
        + "\n\nCompare the candidates in this order: (1) fit to visible species anatomy; (2) whether focus lies on "
        "discriminative bird parts rather than background/corruption; (3) whether prototypes visually support the "
        "candidate rather than merely repeat its label; and (4) whether competing evidence is coherent.\n\n"
        "Return exactly one JSON object with: chosen_state (A or B), adaptation_score (integer -5 meaning AFTER "
        "clearly worse through +5 meaning AFTER clearly better), state_a_evidence_quality (1-5), "
        "state_b_evidence_quality (1-5), focus_comparison, prototype_comparison, comparative_analysis (3-6 "
        "evidence-grounded sentences), and confidence (1-5). If boards are absent, explicitly say the relevant "
        "evidence is not observable. Return no markdown or text outside the JSON object."
    )


def validate_reasoned(payload: Dict) -> Dict:
    cleaned = dict(payload)
    chosen = str(payload.get("chosen_state", "")).upper().strip()
    if chosen not in {"A", "B"}:
        raise ValueError(f"chosen_state must be A or B, received {chosen!r}")
    cleaned["chosen_state"] = chosen
    score = int(payload["adaptation_score"])
    if not -5 <= score <= 5:
        raise ValueError(f"adaptation_score must be in [-5, 5], received {score}")
    cleaned["adaptation_score"] = score
    for key in ("state_a_evidence_quality", "state_b_evidence_quality", "confidence"):
        value = int(payload[key])
        if not 1 <= value <= 5:
            raise ValueError(f"{key} must be in [1, 5], received {value}")
        cleaned[key] = value
    for key in ("focus_comparison", "prototype_comparison", "comparative_analysis"):
        value = str(payload.get(key, "")).strip()
        if not value:
            raise ValueError(f"{key} is empty")
        cleaned[key] = value
    return cleaned


def build_skeptical_prompt(
    record: Dict, order: Tuple[str, str], prior_audit: Dict
) -> str:
    state_lines = []
    for label, prefix in zip(("A", "B"), order):
        role = "BEFORE frozen source" if prefix == "before" else "AFTER continuous ProtoTTA"
        state_lines.append(
            f"STATE {label} is {role} and predicts {record[prefix + '_prediction']}."
        )
    prior_state = str(prior_audit["chosen_state"]).upper()
    prior_reason = str(prior_audit.get("comparative_analysis", "")).strip()
    return (
        "You are the second-stage, label-blind evidence auditor for a prototype bird classifier. "
        "Ground truth and correctness are withheld. A first-stage auditor examined only the corrupted bird "
        "and the two candidate species names; it did not see classifier reasoning boards. Preserve that "
        "independent visual judgment unless the boards provide concrete falsifying evidence.\n\n"
        + "\n".join(state_lines)
        + f"\n\nLOCKED IMAGE-ONLY DECISION: STATE {prior_state}.\n"
        + f"LOCKED IMAGE-ONLY ANALYSIS: {prior_reason}\n\n"
        "You receive three images: IMAGE 1 is the corrupted bird; IMAGE 2 is STATE A's rich reasoning board; "
        "IMAGE 3 is STATE B's rich reasoning board. Each board contains a focus map, five predicted-class "
        "training prototypes, and ten any-class/competing prototypes. A state's board is generated from that "
        "same classifier state, so internally consistent prototypes are expected even when its prediction is wrong. "
        "Higher contribution, greater confidence, and more repeated prototypes are not independent evidence.\n\n"
        "Override the locked image-only decision ONLY when the boards expose at least one visible failure flag:\n"
        "1. BACKGROUND_FOCUS: its focus is mainly corruption, background, or non-bird pixels while the alternative is anatomical;\n"
        "2. REGION_PATCH_MISMATCH: its retrieved prototype patches visibly mismatch the highlighted raw-image region;\n"
        "3. CONTRADICTORY_COMPETITORS: its any-class evidence visibly and specifically supports the other candidate;\n"
        "4. NONDISCRIMINATIVE_FOCUS: it relies on generic anatomy while the alternative localizes a candidate-specific marking.\n"
        "Do not invent a flag merely because the other state is AFTER or looks more confident. If no flag clearly "
        "falsifies the locked decision, keep it.\n\n"
        "Return exactly one JSON object with: chosen_state (A or B), adaptation_score (-5 to +5 for AFTER versus "
        "BEFORE), override_image_decision (true or false), failure_flags (JSON array using only the four names above), "
        "state_a_evidence_quality (1-5), state_b_evidence_quality (1-5), focus_comparison, prototype_comparison, "
        "comparative_analysis (3-6 evidence-grounded sentences), and confidence (1-5). Return no markdown."
    )


def validate_skeptical(payload: Dict, prior_state: str) -> Dict:
    cleaned = validate_reasoned(payload)
    allowed = {
        "BACKGROUND_FOCUS",
        "REGION_PATCH_MISMATCH",
        "CONTRADICTORY_COMPETITORS",
        "NONDISCRIMINATIVE_FOCUS",
    }
    flags = [str(item).upper().strip() for item in payload.get("failure_flags", [])]
    if any(item not in allowed for item in flags):
        raise ValueError(f"invalid failure_flags: {flags}")
    reported_override = payload.get("override_image_decision", False)
    if isinstance(reported_override, bool):
        override = reported_override
    elif isinstance(reported_override, (int, float)) and reported_override in (0, 1):
        override = bool(reported_override)
    elif isinstance(reported_override, str) and reported_override.strip().lower() in {"true", "false"}:
        override = reported_override.strip().lower() == "true"
    else:
        raise ValueError(f"invalid override_image_decision: {reported_override!r}")
    actual_override = cleaned["chosen_state"] != prior_state
    if actual_override and not flags:
        raise ValueError("an override requires at least one visible failure flag")
    cleaned["failure_flags"] = flags
    cleaned["reported_override_image_decision"] = override
    cleaned["override_field_consistent"] = override == actual_override
    cleaned["override_image_decision"] = actual_override
    return cleaned


def score_file(output_dir: Path, condition: str, sample: Dict) -> Path:
    return output_dir / "scores" / condition / sample["corruption"] / f"{sample['public_id']}.json"


def run_score(args: argparse.Namespace) -> None:
    from vlm_eval import VLMScorer

    manifest = read_json(args.output_dir / "manifest.json")
    samples = manifest["samples"][: args.max_samples]
    reasoned = args.condition in REASONED_CONDITIONS
    scorer = VLMScorer(
        args.model_id,
        max_new_tokens=args.max_new_tokens if reasoned else 1,
        enable_thinking=False,
    )
    for index, sample in enumerate(samples, start=1):
        destination = score_file(args.output_dir, args.condition, sample)
        if destination.exists() and not args.overwrite:
            continue
        corruption, key = sample["corruption"], sample["key"]
        export = read_json(full.export_path(args.fullset_dir, corruption))
        record, evidence = export["records"][key], export["evidence"][key]
        reverse = args.condition == "rich_nomsp_reverse"
        order = deterministic_order(corruption, key, reverse=reverse)
        raw = args.data_dir / corruption / "5" / record["image_path"]
        temporary = args.output_dir / "rendered" / args.condition / corruption
        if args.condition.startswith("image_") or args.condition == "reasoned_image_nomsp":
            images = [raw]
        elif args.condition.startswith("compact_"):
            board = temporary / f"{sample['public_id']}.jpg"
            compact_ordered_canvas(
                raw, record, evidence, order, args.prototype_dir, board,
                show_msp=args.condition == "compact_msp",
            )
            images = [board]
        elif args.condition == "candidate_gallery_nomsp":
            board = temporary / f"{sample['public_id']}.jpg"
            candidate_gallery_canvas(raw, record, evidence, order, args.prototype_dir, board)
            images = [board]
        else:
            state_paths = []
            for label, prefix in zip(("A", "B"), order):
                board = temporary / f"{sample['public_id']}_{label}.jpg"
                rich_state_board(raw, record, evidence, prefix, args.prototype_dir, board)
                state_paths.append(board)
            images = [raw, *state_paths]
        prior_audit = None
        if args.condition == "skeptical_rich_nomsp":
            prior_path = score_file(args.output_dir, "reasoned_image_nomsp", sample)
            if not prior_path.exists():
                raise RuntimeError(f"Missing required image-only audit: {prior_path}")
            prior_audit = read_json(prior_path)["reasoned_audit"]
            prompt = build_skeptical_prompt(record, order, prior_audit)
        else:
            prompt = (
                build_reasoned_prompt(record, order, args.condition)
                if reasoned else build_prompt(record, order, args.condition)
            )
        extra = {}
        if reasoned:
            parsed, raw_text, actual_prompt = scorer.generate_json(images, prompt)
            audit = (
                validate_skeptical(parsed, str(prior_audit["chosen_state"]).upper())
                if args.condition == "skeptical_rich_nomsp"
                else validate_reasoned(parsed)
            )
            preferred = audit["chosen_state"]
            preferred_prefix = order[0] if preferred == "A" else order[1]
            adapted_probability = (float(audit["adaptation_score"]) + 5.0) / 10.0
            probability_a = None
            probability_b = None
            extra = {"reasoned_audit": audit, "raw_response": raw_text}
            prompt = actual_prompt
        else:
            preferred, probability_a, probability_b = choice_probability(scorer, images, prompt)
            preferred_prefix = order[0] if preferred == "A" else order[1]
            adapted_probability = probability_a if order[0] == "after" else probability_b
        write_json(
            destination,
            {
                "schema_version": 1,
                "condition": args.condition,
                "corruption": corruption,
                "public_id": sample["public_id"],
                "label_visible_to_vlm": False,
                "state_a_role": order[0],
                "state_b_role": order[1],
                "preferred_state": preferred,
                "preferred_role": preferred_prefix,
                "probability_a": probability_a,
                "probability_b": probability_b,
                "adapted_probability": adapted_probability,
                "model_id": args.model_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                **extra,
            },
        )
        LOGGER.info("%s %d/%d %s p(after)=%.3f", args.condition, index, len(samples), sample["public_id"], adapted_probability)


def bootstrap_accuracy_delta(left: np.ndarray, right: np.ndarray, rng, n: int) -> Tuple[float, float]:
    values = []
    for _ in range(n):
        indices = rng.integers(0, len(left), len(left))
        values.append(float(np.mean(left[indices] - right[indices])))
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def run_analyze(args: argparse.Namespace) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    manifest = read_json(args.output_dir / "manifest.json")
    rows = []
    for sample in manifest["samples"]:
        row = dict(sample)
        row["target_after"] = int(sample["outcome"] == "beneficial")
        for condition in CONDITIONS:
            path = score_file(args.output_dir, condition, sample)
            if not path.exists():
                raise RuntimeError(f"Missing diagnostic score: {path}")
            score = read_json(path)
            row[condition + "_probability"] = float(score["adapted_probability"])
            row[condition + "_correct"] = int((score["preferred_role"] == "after") == bool(row["target_after"]))
        rows.append(row)
    target = np.asarray([row["target_after"] for row in rows])
    summary = {}
    for condition in CONDITIONS:
        probability = np.asarray([row[condition + "_probability"] for row in rows])
        correct = np.asarray([row[condition + "_correct"] for row in rows])
        summary[condition] = {
            "accuracy": float(correct.mean()),
            "auroc": float(roc_auc_score(target, probability)),
            "aupr": float(average_precision_score(target, probability)),
            "accept_rate": float((probability >= 0.5).mean()),
            "n": len(rows),
        }
    rng = np.random.default_rng(args.seed)
    rich = np.asarray([row["rich_nomsp_correct"] for row in rows])
    compact = np.asarray([row["compact_nomsp_correct"] for row in rows])
    reverse = np.asarray([row["rich_nomsp_reverse_correct"] for row in rows])
    reasoned_rich = np.asarray([row["reasoned_rich_nomsp_correct"] for row in rows])
    reasoned_image = np.asarray([row["reasoned_image_nomsp_correct"] for row in rows])
    skeptical = np.asarray([row["skeptical_rich_nomsp_correct"] for row in rows])
    comparisons = {
        "rich_minus_compact_nomsp": {
            "delta": float(np.mean(rich - compact)),
            "bootstrap_95_ci": bootstrap_accuracy_delta(rich, compact, rng, args.bootstrap_resamples),
        },
        "rich_minus_reverse": {
            "delta": float(np.mean(rich - reverse)),
            "bootstrap_95_ci": bootstrap_accuracy_delta(rich, reverse, rng, args.bootstrap_resamples),
        },
        "reasoned_rich_minus_reasoned_image": {
            "delta": float(np.mean(reasoned_rich - reasoned_image)),
            "bootstrap_95_ci": bootstrap_accuracy_delta(
                reasoned_rich, reasoned_image, rng, args.bootstrap_resamples
            ),
        },
        "skeptical_rich_minus_reasoned_image": {
            "delta": float(np.mean(skeptical - reasoned_image)),
            "bootstrap_95_ci": bootstrap_accuracy_delta(
                skeptical, reasoned_image, rng, args.bootstrap_resamples
            ),
        },
    }
    overrides = []
    for sample in manifest["samples"]:
        score = read_json(score_file(args.output_dir, "skeptical_rich_nomsp", sample))
        audit = score.get("reasoned_audit", {})
        if audit.get("override_image_decision"):
            target_after = sample["outcome"] == "beneficial"
            final_after = score["preferred_role"] == "after"
            overrides.append(int(final_after == target_after))
    comparisons["skeptical_override_audit"] = {
        "num_overrides": len(overrides),
        "correct_overrides": int(sum(overrides)),
        "incorrect_overrides": int(len(overrides) - sum(overrides)),
    }
    payload = {"schema_version": 1, "label_visible_to_vlm": False, "summary": summary, "comparisons": comparisons}
    write_json(args.output_dir / "diagnostic_metrics.json", payload)
    lines = [
        "# VLM Gate Input Diagnostics",
        "",
        "Condition | Balanced selection accuracy | AUROC | AUPR | Accept rate",
        "--------- | --------------------------- | ----- | ---- | -----------",
    ]
    for condition in CONDITIONS:
        item = summary[condition]
        lines.append(
            f"{condition} | {item['accuracy']:.3f} | {item['auroc']:.3f} | {item['aupr']:.3f} | {item['accept_rate']:.3f}"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, item in comparisons.items():
        if "bootstrap_95_ci" in item:
            lo, hi = item["bootstrap_95_ci"]
            lines.append(f"- `{name}`: {item['delta']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}].")
        else:
            lines.append(
                f"- `{name}`: {item['correct_overrides']}/{item['num_overrides']} overrides correct "
                f"({item['incorrect_overrides']} incorrect)."
            )
    report = "\n".join(lines) + "\n"
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    for name in ("fullset_dir", "output_dir", "data_dir", "prototype_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "score":
        run_score(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
