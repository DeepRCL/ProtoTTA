#!/usr/bin/env python3
"""Full CUB-200-C evaluation of a VLM gate on top of fixed ProtoTTA.

Stages
------
export
    Replay the exact fixed-lambda=1.0 ProtoTTA configuration from the paper
    table and export sample predictions plus prototype evidence.
score
    Query Qwen3.6 in batches only where BEFORE and AFTER predictions differ.
    Conditions are image+predictions and image+paired reasoning boards.

Ground truth is stored by export for evaluation but never rendered or included
in VLM prompts.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import logging
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "defocus_blur",
    "gaussian_blur",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "jpeg_compression",
    "pixelate",
]
CONDITIONS = ["image_predictions", "full_reasoning"]
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
LOGGER = logging.getLogger("fullset_vlm_gate")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "vlm_fullset_gate")
    parser.add_argument("--data-dir", type=Path, default=root / "datasets" / "cub200_c")
    parser.add_argument("--domain-name", default="bird")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "saved_models" / "deit_small_patch16_224" / "exp1" / "14finetuned0.8609.pth",
    )
    parser.add_argument(
        "--prototype-dir",
        type=Path,
        default=root / "saved_models" / "deit_small_patch16_224" / "exp1" / "img" / "epoch-4",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--corruption", choices=CORRUPTIONS, required=True)
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--batch-size", type=int, default=128)
    export.add_argument("--max-batches", type=int, default=None)
    export.add_argument("--overwrite", action="store_true")

    score = subparsers.add_parser("score")
    score.add_argument("--corruption", choices=CORRUPTIONS, required=True)
    score.add_argument("--condition", choices=CONDITIONS, required=True)
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--vlm-model-id", default=MODEL_ID)
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--max-new-tokens", type=int, default=4096)
    score.add_argument("--max-samples", type=int, default=None)
    score.add_argument("--allow-incomplete-export", action="store_true", help=argparse.SUPPRESS)
    score.add_argument("--overwrite", action="store_true")
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


def export_path(output_dir: Path, corruption: str, seed: int = 0) -> Path:
    return output_dir / "export" / f"seed{seed}" / f"{corruption}.json"


def score_path(output_dir: Path, condition: str, corruption: str) -> Path:
    return output_dir / "scores" / condition / f"{corruption}.json"


def humanize(name: str) -> str:
    value = name.split(".", 1)[-1].replace("_", " ")
    return value[:1].upper() + value[1:]


def stable_key(corruption: str, relative_path: str) -> str:
    return f"{corruption}::{relative_path}"


def public_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def exact_fixed_prototta(base_model):
    """Match evaluate_robustness.py proto_imp_conf_v3, lambda=1.0 flags."""
    import evaluate_robustness as robustness

    robustness.cfg.MODEL.EPISODIC = False
    robustness.cfg.OPTIM.STEPS = 1
    return robustness.setup_proto_entropy(
        base_model,
        use_importance=True,
        use_confidence=True,
        use_geometric_filter=True,
        geo_filter_threshold=0.92,
        consensus_strategy="top_k_mean",
        consensus_ratio=0.5,
        adaptation_mode="layernorm_attn_bias",
        logit_weight=0.0,
        shared_confidence_weighting=True,
        gradient_normalize=True,
        # Match the paper-table run metadata exactly. Although diagnostics are
        # intended to be observational, retaining the identical execution path
        # avoids small trajectory differences from extra CUDA synchronizations.
        record_diagnostics=True,
    )


def _slice_outputs(outputs: Tuple, indices):
    return tuple(value[indices] for value in outputs)


def run_export(args: argparse.Namespace) -> None:
    import torch
    import interpretability_viz as viz
    import vlm_eval

    destination = export_path(args.output_dir, args.corruption, args.seed)
    if destination.exists() and not args.overwrite:
        LOGGER.info("Export already exists: %s", destination)
        return
    vlm_eval.install_timm_checkpoint_compat()
    vlm_eval.lazy_import_runtime_modules()
    import evaluate_robustness as robustness

    robustness.seed_everything(args.seed)
    device = torch.device("cuda")
    dataset = vlm_eval.load_corruption_dataset(args.data_dir, args.corruption, 5)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    source = torch.load(str(args.model), weights_only=False).to(device).eval()
    adapted_base = torch.load(str(args.model), weights_only=False).to(device)
    adapted = exact_fixed_prototta(adapted_base)
    adapted.eval()
    class_names = dataset.classes
    records: Dict[str, Dict] = {}
    evidence: Dict[str, Dict] = {}
    cursor = 0

    for batch_index, (images, labels) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            before_outputs = source(images)
            after_outputs = adapted(images)
        before_logits, _, _ = vlm_eval.normalize_outputs(before_outputs)
        after_logits, _, _ = vlm_eval.normalize_outputs(after_outputs)
        before_pred = before_logits.argmax(dim=1)
        after_pred = after_logits.argmax(dim=1)
        before_prob = torch.softmax(before_logits.float(), dim=1)
        after_prob = torch.softmax(after_logits.float(), dim=1)
        changed_indices = torch.where(before_pred != after_pred)[0]

        before_proto_by_local = {}
        after_proto_by_local = {}
        if len(changed_indices):
            changed_images = images[changed_indices]
            before_batches = viz.get_top_k_prototypes_batch(
                source,
                changed_images,
                k=None,
                precomputed_outputs=_slice_outputs(before_outputs, changed_indices),
                sort_by="activation",
            )
            adapted_net = vlm_eval.get_underlying_model(adapted)
            after_batches = viz.get_top_k_prototypes_batch(
                adapted_net,
                changed_images,
                k=None,
                precomputed_outputs=_slice_outputs(after_outputs, changed_indices),
                sort_by="activation",
            )
            for position, local_tensor in enumerate(changed_indices):
                local = int(local_tensor.item())
                before_all, _ = before_batches[position]
                after_all, _ = after_batches[position]
                before_proto_by_local[local] = (
                    vlm_eval.select_predicted_class_prototypes(before_all, int(before_pred[local]), source, 5),
                    vlm_eval.select_any_class_prototypes(before_all, source, 10),
                )
                after_proto_by_local[local] = (
                    vlm_eval.select_predicted_class_prototypes(after_all, int(after_pred[local]), adapted_net, 5),
                    vlm_eval.select_any_class_prototypes(after_all, adapted_net, 10),
                )

        for local in range(labels.size(0)):
            dataset_index = cursor + local
            absolute_path, _ = dataset.samples[dataset_index]
            relative_path = str(Path(absolute_path).relative_to(dataset.root))
            key = stable_key(args.corruption, relative_path)
            gt = int(labels[local].item())
            bp = int(before_pred[local].item())
            ap = int(after_pred[local].item())
            before_top2 = torch.topk(before_prob[local], k=2).values
            after_top2 = torch.topk(after_prob[local], k=2).values
            records[key] = {
                "sample_index": dataset_index,
                "image_path": relative_path,
                "ground_truth_index": gt,
                "before_prediction_index": bp,
                "after_prediction_index": ap,
                "before_prediction": humanize(class_names[bp]),
                "after_prediction": humanize(class_names[ap]),
                "before_correct": bp == gt,
                "after_correct": ap == gt,
                "before_msp": float(before_top2[0].item()),
                "after_msp": float(after_top2[0].item()),
                "before_margin": float((before_top2[0] - before_top2[1]).item()),
                "after_margin": float((after_top2[0] - after_top2[1]).item()),
            }
            if local in before_proto_by_local:
                before_predicted, before_any = before_proto_by_local[local]
                after_predicted, after_any = after_proto_by_local[local]
                evidence[key] = {
                    "before_predicted": [vlm_eval.proto_to_json(item, class_names) for item in before_predicted],
                    "before_any": [vlm_eval.proto_to_json(item, class_names) for item in before_any],
                    "after_predicted": [vlm_eval.proto_to_json(item, class_names) for item in after_predicted],
                    "after_any": [vlm_eval.proto_to_json(item, class_names) for item in after_any],
                }
        cursor += labels.size(0)
        LOGGER.info(
            "%s batch=%d records=%d disagreements=%d",
            args.corruption,
            batch_index,
            len(records),
            len(evidence),
        )

    payload = {
        "schema_version": 1,
        "configuration": "ProtoViT fixed ProtoTTA lambda=1.0; shared confidence weighting; gradient normalization",
        "seed": args.seed,
        "corruption": args.corruption,
        "severity": 5,
        "complete": len(records) == len(dataset),
        "num_dataset_samples": len(dataset),
        "num_records": len(records),
        "num_disagreements": len(evidence),
        "before_accuracy": float(np.mean([item["before_correct"] for item in records.values()])),
        "after_accuracy": float(np.mean([item["after_correct"] for item in records.values()])),
        "records": records,
        "evidence": evidence,
    }
    write_json(destination, payload)
    LOGGER.info("Wrote %s", destination)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def _square(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def focus_overlay(raw: Image.Image, prototypes: Sequence[Dict], size: int) -> Image.Image:
    base = _square(raw, size)
    if not prototypes:
        return base
    heat = np.zeros((14, 14), dtype=np.float32)
    proto = prototypes[0]
    locations = proto.get("patch_locations", [[], []])
    slots = proto.get("slots", [])
    for index, value in enumerate(slots):
        if value <= 0 or index >= len(locations[0]) or index >= len(locations[1]):
            continue
        row, col = int(locations[0][index]), int(locations[1][index])
        if 0 <= row < 14 and 0 <= col < 14:
            heat[row, col] = 1.0
    heat_image = Image.fromarray(np.uint8(heat * 255), mode="L").resize((size, size), Image.Resampling.BILINEAR)
    alpha = np.asarray(heat_image, dtype=np.float32) / 255.0
    alpha = np.uint8(np.clip(alpha * 210, 0, 210))
    red = Image.new("RGBA", (size, size), (255, 35, 0, 0))
    red.putalpha(Image.fromarray(alpha, mode="L"))
    return Image.alpha_composite(base.convert("RGBA"), red).convert("RGB")


def prototype_image(prototype_dir: Path, proto: Dict, size: Tuple[int, int]) -> Image.Image:
    explicit = proto.get("prototype_image_path")
    if explicit:
        path = Path(str(explicit))
        if path.exists():
            return ImageOps.fit(
                Image.open(path).convert("RGB"),
                size,
                method=Image.Resampling.LANCZOS,
            )
    import vlm_eval

    patch = vlm_eval.load_prototype_patch(prototype_dir, int(proto["proto_idx"]))
    if patch is None:
        return Image.new("RGB", size, "#EEEEEE")
    array = np.asarray(patch)
    if array.dtype != np.uint8:
        array = np.uint8(np.clip(array[..., :3], 0, 1) * 255)
    elif array.shape[-1] > 3:
        array = array[..., :3]
    image = Image.fromarray(array).convert("RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def _draw_prototype_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    width: int,
    title: str,
    prototypes: Sequence[Dict],
    prototype_dir: Path,
) -> None:
    draw.text((x0 + 20, y0), title, fill="#222222", font=font(23, True))
    y = y0 + 35
    card_width = 250
    gap = 22
    for index, proto in enumerate(prototypes[:3]):
        x = x0 + 20 + index * (card_width + gap)
        patch = prototype_image(prototype_dir, proto, (card_width, 145))
        canvas.paste(patch, (x, y))
        label = textwrap.shorten(str(proto.get("class_name", "prototype")), width=22, placeholder="...")
        detail = f"{label} | P{proto['proto_idx']} | {float(proto.get('contribution', 0)):.2f}"
        draw.text((x, y + 151), detail, fill="#222222", font=font(16))


def compact_evidence_canvas(
    raw_path: Path,
    record: Dict,
    evidence: Dict,
    prototype_dir: Path,
    destination: Path,
) -> None:
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGB")
    side_width, height, gutter = 860, 1030, 24
    canvas = Image.new("RGB", (2 * side_width + gutter, height), "white")
    draw = ImageDraw.Draw(canvas)
    roles = [
        ("BEFORE: FROZEN SOURCE", "before", "#285A8E"),
        ("AFTER: CONTINUOUS PROTOTTA", "after", "#A54822"),
    ]
    for column, (role, prefix, color) in enumerate(roles):
        x0 = column * (side_width + gutter)
        draw.rectangle((x0, 0, x0 + side_width, 120), fill=color)
        draw.text((x0 + 20, 15), role, fill="white", font=font(30, True))
        prediction = record[f"{prefix}_prediction"]
        msp = record[f"{prefix}_msp"]
        draw.text((x0 + 20, 61), f"Prediction: {prediction} | MSP {msp:.3f}", fill="white", font=font(23, True))
        predicted = evidence[f"{prefix}_predicted"]
        any_class = evidence[f"{prefix}_any"]
        competitors = [item for item in any_class if int(item["class_index"]) != int(record[f"{prefix}_prediction_index"])]
        if len(competitors) < 3:
            competitors = list(any_class)
        raw_square = _square(raw, 300)
        focus = focus_overlay(raw, predicted, 300)
        canvas.paste(raw_square, (x0 + 60, 145))
        canvas.paste(focus, (x0 + 500, 145))
        draw.text((x0 + 60, 452), "Corrupted input", fill="#222222", font=font(20, True))
        draw.text((x0 + 500, 452), "Prototype focus", fill="#222222", font=font(20, True))
        _draw_prototype_row(canvas, draw, x0, 495, side_width, "Top predicted-class prototypes", predicted, prototype_dir)
        _draw_prototype_row(canvas, draw, x0, 755, side_width, "Strong competing-class prototypes", competitors, prototype_dir)
    draw.rectangle((side_width, 0, side_width + gutter, height), fill="#222222")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90, optimize=True)


def build_batch_prompt(
    condition: str,
    batch: Sequence[Tuple[str, Dict]],
    domain_name: str = "bird",
    compact_response: bool = False,
) -> str:
    lines = [
        f"You are a label-free supervisor for a prototype-based {domain_name} classifier.",
        "For every sample, decide whether to ACCEPT its AFTER continuous ProtoTTA prediction or ROLLBACK to BEFORE.",
        "BEFORE is the frozen source model. AFTER accumulated online updates from preceding batches in the corruption stream and was not reset per image.",
        "Adaptation may help or harm. Ground truth and correctness are withheld. Do not assume AFTER is better.",
        "",
    ]
    if condition == "image_predictions":
        lines.append(
            f"Each image is the corrupted test {domain_name} image. "
            "No prototype reasoning board is supplied."
        )
    else:
        lines.append(
            "Each image is a paired reasoning canvas: blue/left is BEFORE and orange/right is AFTER. "
            "Compare localization, predicted-class prototype match, competing prototypes, and confidence."
        )
    lines.extend(["", "Samples and images are aligned in this exact order:"])
    for index, (identifier, record) in enumerate(batch, start=1):
        lines.append(
            f"IMAGE {index} / ID {identifier}: BEFORE predicts {record['before_prediction']} (MSP {record['before_msp']:.3f}); "
            f"AFTER predicts {record['after_prediction']} (MSP {record['after_msp']:.3f})."
        )
    if compact_response:
        required_ids = ", ".join(identifier for identifier, _ in batch)
        lines.extend([
            "",
            "Return exactly one compact JSON object with key 'scores'. Map every ID to one integer from -5 to +5, in the same order.",
            f"The scores object must contain exactly these {len(batch)} keys, with none omitted: {required_ids}.",
            "A negative score means ROLLBACK is more appropriate; a positive score means ACCEPT is more appropriate. Return no explanations, markdown, or other keys.",
        ])
    else:
        lines.extend([
            "",
            "Return exactly one JSON object with key 'decisions'. Its value must be an array containing one object per ID, in the same order.",
            "Each decision object must contain: id, action (exactly ACCEPT or ROLLBACK), "
            "adaptation_score (integer -5 to +5), and rationale (one concise evidence-grounded sentence).",
            "ACCEPT only when AFTER is more likely correct than BEFORE. Otherwise ROLLBACK. Return no markdown or text outside JSON.",
        ])
    return "\n".join(lines)


def validate_batch_response(
    payload: Dict,
    identifiers: Sequence[str],
    compact_response: bool = False,
) -> Dict[str, Dict]:
    if compact_response:
        scores = payload.get("scores")
        if scores is None and isinstance(payload, dict):
            # Qwen may omit the requested wrapper for this deliberately terse
            # schema and return the ID-to-score mapping directly.
            scores = payload
        if isinstance(scores, list):
            scores = dict(zip(identifiers, scores))
        elif isinstance(scores, (int, float)) and len(identifiers) == 1:
            scores = {identifiers[0]: scores}
        if isinstance(scores, dict):
            normalized_scores = {}
            for raw_identifier, value in scores.items():
                match = re.fullmatch(r"S0*(\d+)", str(raw_identifier).strip(), re.IGNORECASE)
                identifier = f"S{int(match.group(1)):02d}" if match else str(raw_identifier)
                normalized_scores[identifier] = value
            scores = normalized_scores
        if (
            not isinstance(scores, dict)
            or not scores
            or not set(scores).issubset(set(identifiers))
        ):
            raise ValueError("VLM compact response contains missing or unexpected score IDs")
        output = {}
        for identifier in identifiers:
            if identifier not in scores:
                continue
            score = int(scores[identifier])
            if not -5 <= score <= 5:
                raise ValueError(f"Invalid adaptation_score for {identifier}: {score}")
            output[identifier] = {
                "action": "ROLLBACK" if score < 0 else "ACCEPT",
                "adaptation_score": score,
                "rationale": "score_only",
            }
        return output
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("VLM response must contain a decisions array")
    output = {}
    for decision in decisions:
        identifier = str(decision.get("id", ""))
        if identifier not in identifiers or identifier in output:
            raise ValueError(f"Unexpected or duplicate decision id: {identifier}")
        action = str(decision.get("action", "")).upper().strip()
        if action not in {"ACCEPT", "ROLLBACK"}:
            raise ValueError(f"Invalid action for {identifier}: {action}")
        score = int(decision["adaptation_score"])
        if not -5 <= score <= 5:
            raise ValueError(f"Invalid adaptation_score for {identifier}: {score}")
        rationale = str(decision.get("rationale", "")).strip()
        if not rationale:
            raise ValueError(f"Empty rationale for {identifier}")
        output[identifier] = {"action": action, "adaptation_score": score, "rationale": rationale}
    if set(output) != set(identifiers):
        raise ValueError(f"Missing decisions: {set(identifiers) - set(output)}")
    return output


def chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_score(args: argparse.Namespace) -> None:
    from vlm_eval import VLMScorer

    exported = read_json(export_path(args.output_dir, args.corruption, args.seed))
    if not exported.get("complete") and not args.allow_incomplete_export:
        raise RuntimeError(f"Export is not complete for {args.corruption}")
    destination = score_path(args.output_dir, args.condition, args.corruption)
    if destination.exists() and not args.overwrite:
        result = read_json(destination)
    else:
        result = {
            "schema_version": 1,
            "condition": args.condition,
            "corruption": args.corruption,
            "seed": args.seed,
            "vlm_model_id": args.vlm_model_id,
            "thinking_enabled": False,
            "ground_truth_visible_to_vlm": False,
            "entries": {},
        }
    pending = [key for key in exported["evidence"] if key not in result["entries"]]
    if args.max_samples is not None:
        pending = pending[: args.max_samples]
    scorer = VLMScorer(args.vlm_model_id, args.max_new_tokens, enable_thinking=False)
    records = exported["records"]

    # Long full-set runs inevitably encounter an occasional malformed model
    # response.  Keep every validated checkpoint and recover by subdividing only
    # the failed batch.  This also avoids throwing away hours of valid decisions
    # because one 12-image response used a malformed ID or wrapper.
    work = deque([list(batch) for batch in chunks(pending, args.batch_size)])
    batch_number = 0
    recovery_events = list(result.get("recovery_events", []))
    singleton_attempts: Dict[str, int] = {}
    while work:
        keys = work.popleft()
        batch_number += 1
        batch = [(f"S{index:02d}", records[key]) for index, key in enumerate(keys, start=1)]
        # Score-only responses reduce decoding cost without changing the
        # evidence supplied to the supervisor or the downstream gate inputs.
        compact_response = args.condition == "full_reasoning"
        prompt = build_batch_prompt(
            args.condition,
            batch,
            args.domain_name,
            compact_response=compact_response,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="vlm_gate_") as temporary:
                temporary_path = Path(temporary)
                images = []
                for index, key in enumerate(keys, start=1):
                    record = records[key]
                    raw = args.data_dir / args.corruption / "5" / record["image_path"]
                    if args.condition == "image_predictions":
                        images.append(raw)
                    else:
                        canvas = temporary_path / f"S{index:02d}.jpg"
                        compact_evidence_canvas(
                            raw,
                            record,
                            exported["evidence"][key],
                            args.prototype_dir,
                            canvas,
                        )
                        images.append(canvas)
                parsed, raw_text, actual_prompt = scorer.generate_json(images, prompt)
                identifiers = [item[0] for item in batch]
                decisions = validate_batch_response(
                    parsed,
                    identifiers,
                    compact_response=compact_response,
                )
        except Exception as exc:
            event = {
                "batch_size": len(keys),
                "first_public_id": public_id(keys[0]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if "raw_text" in locals():
                event["raw_response"] = str(raw_text)[-2000:]
            recovery_events.append(event)
            result["recovery_events"] = recovery_events
            write_json(destination, result)
            if len(keys) > 1:
                midpoint = len(keys) // 2
                LOGGER.warning(
                    "%s/%s validation failed for %d samples; retrying as %d + %d: %s",
                    args.corruption,
                    args.condition,
                    len(keys),
                    midpoint,
                    len(keys) - midpoint,
                    exc,
                )
                work.appendleft(keys[midpoint:])
                work.appendleft(keys[:midpoint])
                continue
            key = keys[0]
            singleton_attempts[key] = singleton_attempts.get(key, 0) + 1
            if singleton_attempts[key] < 5:
                LOGGER.warning(
                    "%s/%s singleton %s failed attempt %d/5; retrying: %s",
                    args.corruption,
                    args.condition,
                    public_id(key),
                    singleton_attempts[key],
                    exc,
                )
                work.appendleft(keys)
                continue
            raise RuntimeError(
                f"VLM failed five times for singleton {public_id(key)}"
            ) from exc
        key_by_identifier = dict(zip(identifiers, keys))
        for identifier, decision in decisions.items():
            key = key_by_identifier[identifier]
            result["entries"][key] = {
                "public_id": public_id(key),
                **decision,
            }
        if compact_response and len(decisions) < len(identifiers):
            omitted = [
                key_by_identifier[identifier]
                for identifier in identifiers
                if identifier not in decisions
            ]
            recovery_events.append({
                "batch_size": len(keys),
                "first_public_id": public_id(keys[0]),
                "partial_compact_response": len(decisions),
                "omitted_for_retry": len(omitted),
            })
            work.appendleft(omitted)
        result["num_expected"] = len(exported["evidence"])
        result["num_complete"] = len(result["entries"])
        result["complete"] = len(result["entries"]) == len(exported["evidence"])
        result["response_format"] = (
            "compact_score_only" if compact_response else "concise_sentence"
        )
        result["recovery_events"] = recovery_events
        result["last_prompt_sha256"] = hashlib.sha256(actual_prompt.encode("utf-8")).hexdigest()
        result["last_raw_response"] = raw_text
        write_json(destination, result)
        LOGGER.info(
            "%s/%s batch=%d complete=%d/%d",
            args.corruption,
            args.condition,
            batch_number,
            len(result["entries"]),
            len(exported["evidence"]),
        )


def development_keys(path: Path) -> set:
    if not path.exists():
        return set()
    manifest = read_json(path)
    return {
        stable_key(str(sample["corruption_type"]), str(sample["image_path"]))
        for sample in manifest.get("samples", [])
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    for name in ("output_dir", "data_dir", "model", "prototype_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == "export":
        run_export(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
