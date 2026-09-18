#!/usr/bin/env python3
"""Full CUB-200-C evaluation of a VLM gate on top of fixed ProtoTTA.

Stages
------
export
    Replay the exact fixed-lambda=1.0 ProtoTTA configuration from the paper
    table and export sample predictions plus prototype evidence.
score
    Query Qwen3.6 in batches only where BEFORE and AFTER predictions differ.
    Matched conditions are image+predictions and image+paired reasoning boards.
analyze
    Apply ACCEPT/ROLLBACK decisions to every test prediction and report full-set
    and prompt-development-excluded accuracy by corruption.

Ground truth is stored by export for evaluation but never rendered or included
in VLM prompts.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import logging
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
CONDITION_NAMES = {
    "image_predictions": "Image + predictions/confidence",
    "full_reasoning": "Image + predictions/confidence + paired reasoning boards",
}
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
PAPER_SEEDS = [0, 2, 3]
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

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "--development-subset",
        type=Path,
        default=root / "results" / "vlm_eval" / "subset.json",
    )
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
        reset_mode=None,
        reset_frequency=10,
        confidence_threshold=0.7,
        ema_alpha=0.999,
        use_geometric_filter=True,
        geo_filter_threshold=0.92,
        consensus_strategy="top_k_mean",
        consensus_ratio=0.5,
        adaptation_mode="layernorm_attn_bias",
        use_ensemble_entropy=False,
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
    lines.extend(
        [
            "",
            "Return exactly one JSON object with key 'decisions'. Its value must be an array containing one object per ID, in the same order.",
            "Each decision object must contain: id, action (exactly ACCEPT or ROLLBACK), adaptation_score (integer -5 to +5), and rationale (one concise evidence-grounded sentence).",
            "ACCEPT only when AFTER is more likely correct than BEFORE. Otherwise ROLLBACK. Return no markdown or text outside JSON.",
        ]
    )
    return "\n".join(lines)


def validate_batch_response(payload: Dict, identifiers: Sequence[str]) -> Dict[str, Dict]:
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
        prompt = build_batch_prompt(args.condition, batch, args.domain_name)
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
                decisions = validate_batch_response(parsed, identifiers)
        except Exception as exc:
            event = {
                "batch_size": len(keys),
                "first_public_id": public_id(keys[0]),
                "error": f"{type(exc).__name__}: {exc}",
            }
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
        for identifier, key in zip(identifiers, keys):
            result["entries"][key] = {
                "public_id": public_id(key),
                **decisions[identifier],
            }
        result["num_expected"] = len(exported["evidence"])
        result["num_complete"] = len(result["entries"])
        result["complete"] = len(result["entries"]) == len(exported["evidence"])
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


def paired_ttest(left: Sequence[float], right: Sequence[float]) -> Dict:
    from scipy.stats import ttest_rel

    statistic, pvalue = ttest_rel(left, right)
    delta = np.asarray(left) - np.asarray(right)
    return {
        "mean_difference": float(np.mean(delta)),
        "t_statistic": float(statistic),
        "p_value": float(pvalue),
    }


def evaluate_split(
    exports: Dict[str, Dict],
    scores: Dict[str, Dict[str, Dict]],
    excluded: set,
) -> Dict:
    methods = ["unadapted", "prototta", *CONDITIONS, "oracle"]
    per_corruption = {method: {} for method in methods}
    counts = {method: {} for method in methods}
    action_counts = {condition: {"ACCEPT": 0, "ROLLBACK": 0} for condition in CONDITIONS}
    diagnostics = {
        condition: {
            "beneficial_updates": 0,
            "beneficial_updates_accepted": 0,
            "harmful_updates": 0,
            "harmful_updates_rolled_back": 0,
            "both_wrong_disagreements": 0,
        }
        for condition in CONDITIONS
    }
    for corruption in CORRUPTIONS:
        records = exports[corruption]["records"]
        selected = [(key, record) for key, record in records.items() if key not in excluded]
        correct = {method: 0 for method in methods}
        for key, record in selected:
            correct["unadapted"] += int(record["before_correct"])
            correct["prototta"] += int(record["after_correct"])
            correct["oracle"] += int(record["before_correct"] or record["after_correct"])
            changed = record["before_prediction_index"] != record["after_prediction_index"]
            for condition in CONDITIONS:
                action = scores[condition][corruption]["entries"][key]["action"] if changed else "ACCEPT"
                if changed:
                    action_counts[condition][action] += 1
                    before_correct = bool(record["before_correct"])
                    after_correct = bool(record["after_correct"])
                    diagnostic = diagnostics[condition]
                    if after_correct and not before_correct:
                        diagnostic["beneficial_updates"] += 1
                        diagnostic["beneficial_updates_accepted"] += int(action == "ACCEPT")
                    elif before_correct and not after_correct:
                        diagnostic["harmful_updates"] += 1
                        diagnostic["harmful_updates_rolled_back"] += int(action == "ROLLBACK")
                    else:
                        diagnostic["both_wrong_disagreements"] += 1
                correct[condition] += int(record["after_correct"] if action == "ACCEPT" else record["before_correct"])
        for method in methods:
            per_corruption[method][corruption] = correct[method] / len(selected)
            counts[method][corruption] = {"correct": correct[method], "n": len(selected)}
    summary = {
        method: {
            "accuracy": float(np.mean(list(per_corruption[method].values()))),
            "per_corruption": per_corruption[method],
            "correct": int(sum(item["correct"] for item in counts[method].values())),
            "n": int(sum(item["n"] for item in counts[method].values())),
        }
        for method in methods
    }
    for diagnostic in diagnostics.values():
        beneficial = diagnostic["beneficial_updates"]
        harmful = diagnostic["harmful_updates"]
        diagnostic["beneficial_accept_rate"] = diagnostic["beneficial_updates_accepted"] / beneficial
        diagnostic["harmful_rollback_rate"] = diagnostic["harmful_updates_rolled_back"] / harmful
        diagnostic["decisive_selection_accuracy"] = (
            diagnostic["beneficial_updates_accepted"] + diagnostic["harmful_updates_rolled_back"]
        ) / (beneficial + harmful)
    return {
        "summary": summary,
        "actions_on_disagreements": action_counts,
        "decision_diagnostics": diagnostics,
        "paired_tests": {
            "boards_vs_prototta": paired_ttest(
                list(per_corruption["full_reasoning"].values()),
                list(per_corruption["prototta"].values()),
            ),
            "boards_vs_image_only": paired_ttest(
                list(per_corruption["full_reasoning"].values()),
                list(per_corruption["image_predictions"].values()),
            ),
        },
    }


def report_markdown(payload: Dict) -> str:
    primary = payload["held_out_primary"]
    full = payload["full_secondary"]
    reference = payload["three_seed_reference_verification"]
    labels = {
        "unadapted": "Unadapted",
        "prototta": "Fixed ProtoTTA",
        "image_predictions": "VLM gate without boards",
        "full_reasoning": "VLM gate with boards",
        "oracle": "Oracle before/after",
    }
    lines = [
        "# Full-set VLM-Gated ProtoTTA",
        "",
        "Qwen3.6-35B-A3B was used in deterministic non-thinking mode. Ground truth was hidden during every VLM decision. The gate was queried only when source and fixed-ProtoTTA predictions differed.",
        "",
        "## Primary: prompt-development samples excluded",
        "",
        "Method | Accuracy | Correct / N",
        "------ | -------- | -----------",
    ]
    for method, label in labels.items():
        item = primary["summary"][method]
        lines.append(f"{label} | {100*item['accuracy']:.2f}% | {item['correct']} / {item['n']}")
    lines.extend(["", "## Secondary: complete CUB-200-C test set", "", "Method | Accuracy | Correct / N", "------ | -------- | -----------"])
    for method, label in labels.items():
        item = full["summary"][method]
        lines.append(f"{label} | {100*item['accuracy']:.2f}% | {item['correct']} / {item['n']}")
    lines.extend(
        [
            "",
            "## Primary per-corruption comparison",
            "",
            "Corruption | ProtoTTA | No boards | Boards | Boards - ProtoTTA | Boards - No boards",
            "---------- | -------- | --------- | ------ | ------------------ | ------------------",
        ]
    )
    for corruption in CORRUPTIONS:
        prototta = primary["summary"]["prototta"]["per_corruption"][corruption]
        image_only = primary["summary"]["image_predictions"]["per_corruption"][corruption]
        boards = primary["summary"]["full_reasoning"]["per_corruption"][corruption]
        lines.append(
            f"{corruption} | {100*prototta:.2f}% | {100*image_only:.2f}% | {100*boards:.2f}% | "
            f"{100*(boards-prototta):+.2f} | {100*(boards-image_only):+.2f}"
        )
    lines.extend(
        [
            "",
            "## Label-revealed decision diagnostics (primary, post-hoc only)",
            "",
            "Condition | Accept beneficial | Roll back harmful | Decisive selection accuracy",
            "--------- | ----------------- | ----------------- | ---------------------------",
        ]
    )
    for condition in CONDITIONS:
        item = primary["decision_diagnostics"][condition]
        lines.append(
            f"{CONDITION_NAMES[condition]} | {100*item['beneficial_accept_rate']:.2f}% | "
            f"{100*item['harmful_rollback_rate']:.2f}% | {100*item['decisive_selection_accuracy']:.2f}%"
        )
    lines.extend(["", "## Paired corruption-level tests (primary)", ""])
    for name, item in primary["paired_tests"].items():
        lines.append(f"- `{name}`: delta={100*item['mean_difference']:+.2f} points, paired t-test p={item['p_value']:.6f}.")
    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            f"- Paper seeds: {PAPER_SEEDS}. The existing fixed-ProtoTTA runs are deterministic and identical across these seeds; the same label-blind VLM decisions therefore apply to all three.",
            f"- The fresh evidence replay differs from the saved paper ProtoTTA accuracies by at most {100*reference['max_abs_accuracy_difference']:.3f} percentage points ({reference['max_abs_count_difference']} images). All gate comparisons use the paired fresh replay predictions.",
            f"- Prompt-development exclusions: {payload['num_development_samples']} method-sample cases.",
            "- Both VLM conditions receive the corrupted image, before/after predictions, MSP values, and identical temporal instructions. Only the paired prototype reasoning evidence differs.",
            "- Unchanged predictions default to AFTER because accepting or rolling back cannot alter their predicted label.",
        ]
    )
    return "\n".join(lines) + "\n"


def verify_three_seed_reference(exports: Dict[str, Dict]) -> Dict:
    """Verify the replay against the three fixed-ProtoTTA table result files."""
    root = Path(__file__).resolve().parent
    paths = {
        0: root / "results" / "fixed_lambda_metrics" / "10819" / "cub200c_fixed_lambda1.0_seed0.json",
        2: root / "results" / "fixed_lambda_metrics" / "10819" / "cub200c_fixed_lambda1.0_seed2.json",
        3: root / "results" / "fourth_seed_metrics" / "10921" / "fixed_lambda1.0_seed3.json",
    }
    values = {}
    for seed, path in paths.items():
        document = read_json(path)["results"]["proto_imp_conf_v3"]
        values[seed] = {corruption: float(document[corruption]["5"]["accuracy"]) for corruption in CORRUPTIONS}
    identical = values[0] == values[2] == values[3]
    if not identical:
        raise RuntimeError("The three fixed-ProtoTTA reference runs are not identical")
    differences = {
        corruption: float(exports[corruption]["after_accuracy"] - values[0][corruption])
        for corruption in CORRUPTIONS
    }
    count_differences = {
        corruption: int(round(differences[corruption] * len(exports[corruption]["records"])))
        for corruption in CORRUPTIONS
    }
    max_abs_difference = max(abs(value) for value in differences.values())
    return {
        "paths": {str(seed): str(path) for seed, path in paths.items()},
        "identical_across_seeds": identical,
        "exact_replay_match": max_abs_difference <= 1e-12,
        "max_abs_accuracy_difference": max_abs_difference,
        "max_abs_count_difference": max(abs(value) for value in count_differences.values()),
        "replay_minus_reference": differences,
        "replay_minus_reference_counts": count_differences,
    }


def run_analyze(args: argparse.Namespace) -> None:
    exports = {corruption: read_json(export_path(args.output_dir, corruption)) for corruption in CORRUPTIONS}
    scores = {
        condition: {
            corruption: read_json(score_path(args.output_dir, condition, corruption))
            for corruption in CORRUPTIONS
        }
        for condition in CONDITIONS
    }
    for corruption, export in exports.items():
        if not export.get("complete"):
            raise RuntimeError(f"Incomplete export: {corruption}")
        expected = set(export["evidence"])
        for condition in CONDITIONS:
            available = set(scores[condition][corruption].get("entries", {}))
            if available != expected:
                raise RuntimeError(f"Incomplete scores: {condition}/{corruption} {len(available)}/{len(expected)}")
    dev = development_keys(args.development_subset)
    payload = {
        "schema_version": 1,
        "vlm_model_id": MODEL_ID,
        "thinking_enabled": False,
        "ground_truth_visible_to_vlm": False,
        "paper_seeds": PAPER_SEEDS,
        "seed_runs_identical": True,
        "three_seed_reference_verification": verify_three_seed_reference(exports),
        "num_development_samples": len(dev),
        "held_out_primary": evaluate_split(exports, scores, dev),
        "full_secondary": evaluate_split(exports, scores, set()),
    }
    write_json(args.output_dir / "fullset_vlm_gate_metrics.json", payload)
    markdown = report_markdown(payload)
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    for name in ("output_dir", "data_dir", "model", "prototype_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == "export":
        run_export(args)
    elif args.command == "score":
        run_score(args)
    else:
        args.development_subset = args.development_subset.resolve()
        run_analyze(args)


if __name__ == "__main__":
    main()
