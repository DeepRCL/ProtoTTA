"""Export label-free BEFORE/AFTER examples and ProtoPNet reasoning boards.

One invocation handles exactly one corruption stream.  The source and
ProtoTTA models are freshly loaded, and ProtoTTA is continuous for the whole
corruption.  Its returned logits and captured spatial distances come from the
same pre-update forward used by the normal adaptation wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from proto_baseline.utils.receptive_field import (
    compute_rf_protoL_at_spatial_location,
)

from .evaluate_robustness import (
    STREAM_ORDER_ALGORITHM,
    fixed_stream_indices,
    load_model,
    set_random_seed,
    setup_proto_samplewise_adaptive,
)
from .preprocess import mean, std
from .settings import img_size
from .vlm_prototta_common import (
    CLASS_NAMES,
    CORRUPTIONS,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_lines,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth"
DEFAULT_DATA = ROOT / "datasets/SICAPv2-C"
DEFAULT_PROTOTYPES = ROOT / "saved_models/vgg19_bn/sicapv2_002/img/epoch-20"
METHOD = "ProtoAbsoluteConsistencyCoverageRouter"
SOURCE_METHOD = "Normal"


class SpatialDistanceCapture:
    """Capture exact full spatial distances without adding a model forward."""

    def __init__(self, model: torch.nn.Module):
        self.core = model.core
        self.original = self.core.prototype_distances
        self.value: torch.Tensor | None = None

        def observed(images: torch.Tensor) -> torch.Tensor:
            distances = self.original(images)
            self.value = distances.detach()
            return distances

        self.core.prototype_distances = observed

    def pop(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("model forward did not produce spatial distances")
        result = self.value
        self.value = None
        return result

    def close(self) -> None:
        self.core.prototype_distances = self.original


def _last_layer(model: torch.nn.Module) -> torch.nn.Linear:
    return model.core.last_layer


def _activation_maps(model: torch.nn.Module, distances: torch.Tensor) -> torch.Tensor:
    return model.core.distance_2_similarity(distances)


def _prototype_rows(
    model: torch.nn.Module,
    activation_maps: torch.Tensor,
    predicted_class: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank predicted and competing prototypes by the specified contribution."""
    maps = activation_maps.detach()
    activations = maps.flatten(1).max(dim=1).values
    identities = model.core.prototype_class_identity.argmax(dim=1)
    weights = _last_layer(model).weight.detach()

    predicted_ids = torch.where(identities == predicted_class)[0]
    predicted_contribution = activations[predicted_ids] * weights[
        predicted_class, predicted_ids
    ]
    predicted_order = predicted_ids[
        torch.argsort(predicted_contribution, descending=True)
    ][:3]

    competing_ids = torch.where(identities != predicted_class)[0]
    associated_classes = identities[competing_ids]
    competing_contribution = activations[competing_ids] * weights[
        associated_classes, competing_ids
    ]
    competing_order = competing_ids[
        torch.argsort(competing_contribution, descending=True)
    ][:3]

    def row(proto_id: int, associated_class: int) -> dict[str, Any]:
        proto_map = maps[proto_id]
        flat_location = int(proto_map.argmax().item())
        height, width = proto_map.shape
        y, x = divmod(flat_location, width)
        activation = float(activations[proto_id].item())
        weight = float(weights[associated_class, proto_id].item())
        return {
            "prototype_id": int(proto_id),
            "prototype_class": CLASS_NAMES[associated_class],
            "activation": activation,
            "logit_weight": weight,
            "contribution": activation * weight,
            "focus_map": proto_map.float().cpu().numpy(),
            "focus_location": [int(y), int(x)],
            "focus_map_shape": [int(height), int(width)],
        }

    predicted = [row(int(p), predicted_class) for p in predicted_order]
    competing = [row(int(p), int(identities[p].item())) for p in competing_order]
    if len(predicted) != 3 or len(competing) != 3:
        raise RuntimeError("expected three predicted and three competing prototypes")
    return predicted, competing


def _normalized_heatmap(rows: list[dict[str, Any]], size: tuple[int, int]) -> Image.Image:
    raw = np.maximum.reduce([entry["focus_map"] for entry in rows]).astype(np.float32)
    low, high = float(raw.min()), float(raw.max())
    normalized = np.zeros_like(raw) if high <= low else (raw - low) / (high - low)
    red = normalized
    blue = 1.0 - normalized
    green = np.clip(1.0 - np.abs(normalized - 0.5) * 2.0, 0.0, 1.0) * 0.65
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).resize(
        size, Image.Resampling.BILINEAR
    )


def _focus_overlay(
    image: Image.Image,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
) -> Image.Image:
    base = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    heat = _normalized_heatmap(rows, base.size)
    overlay = Image.blend(base, heat, 0.42)
    strongest = rows[0]
    y, x = strongest["focus_location"]
    bounds = compute_rf_protoL_at_spatial_location(
        img_size, y, x, model.core.proto_layer_rf_info
    )
    draw = ImageDraw.Draw(overlay)
    y0, y1, x0, x1 = bounds
    draw.rectangle((x0, y0, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)),
                   outline=(255, 255, 255), width=3)
    return overlay


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _patch_card(
    prototype_dir: Path,
    row: dict[str, Any],
    width: int = 185,
    height: int = 138,
) -> Image.Image:
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    proto_id = row["prototype_id"]
    patch_path = prototype_dir / f"prototype-img{proto_id}.png"
    if not patch_path.is_file():
        raise FileNotFoundError(f"missing projected prototype patch: {patch_path}")
    with Image.open(patch_path) as source:
        patch = source.convert("RGB")
    patch.thumbnail((width - 10, 82), Image.Resampling.LANCZOS)
    card.paste(patch, ((width - patch.width) // 2, 4))
    label = (
        f"P{proto_id} [{row['prototype_class']}]\n"
        f"contribution {row['contribution']:+.3f}"
    )
    draw.multiline_text((6, 90), label, fill="black", font=_font(13), spacing=2)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(145, 145, 145), width=1)
    return card


def _draw_side(
    canvas: Image.Image,
    x_origin: int,
    image: Image.Image,
    title: str,
    color: tuple[int, int, int],
    predicted_class: str,
    msp: float,
    predicted_rows: list[dict[str, Any]],
    competing_rows: list[dict[str, Any]],
    model: torch.nn.Module,
    prototype_dir: Path,
) -> None:
    draw = ImageDraw.Draw(canvas)
    side_width = 640
    draw.rectangle((x_origin, 0, x_origin + side_width - 1, 718), fill=(248, 248, 248))
    draw.rectangle((x_origin, 0, x_origin + side_width - 1, 52), fill=color)
    draw.text((x_origin + 18, 12), title, fill="white", font=_font(24, bold=True))
    draw.text(
        (x_origin + 18, 60),
        f"Prediction: {predicted_class}     MSP: {msp:.3f}",
        fill=(30, 30, 30), font=_font(18, bold=True),
    )
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    overlay = _focus_overlay(image, predicted_rows, model)
    canvas.paste(resized, (x_origin + 18, 92))
    canvas.paste(overlay, (x_origin + 270, 92))
    draw.text((x_origin + 18, 320), "Same corrupted input", fill="black", font=_font(14))
    draw.text((x_origin + 270, 320), "Actual prototype focus overlay", fill="black", font=_font(14))
    draw.text((x_origin + 18, 346), "Top predicted-class prototypes", fill=color, font=_font(16, bold=True))
    for index, row in enumerate(predicted_rows):
        canvas.paste(_patch_card(prototype_dir, row), (x_origin + 18 + 201 * index, 370))
    draw.text((x_origin + 18, 515), "Top competing-class prototypes", fill=color, font=_font(16, bold=True))
    for index, row in enumerate(competing_rows):
        canvas.paste(_patch_card(prototype_dir, row), (x_origin + 18 + 201 * index, 540))


def render_board(
    destination: Path,
    corrupted_image: Image.Image,
    before: dict[str, Any],
    after: dict[str, Any],
    source_model: torch.nn.Module,
    adapted_base_model: torch.nn.Module,
    prototype_dir: Path,
) -> None:
    canvas = Image.new("RGB", (1280, 720), "white")
    _draw_side(
        canvas, 0, corrupted_image, "BEFORE: FROZEN SOURCE", (37, 99, 176),
        before["class"], before["msp"], before["predicted"], before["competing"],
        source_model, prototype_dir,
    )
    _draw_side(
        canvas, 640, corrupted_image, "AFTER: CONTINUOUS PROTOTTA", (221, 116, 34),
        after["class"], after["msp"], after["predicted"], after["competing"],
        adapted_base_model, prototype_dir,
    )
    draw = ImageDraw.Draw(canvas)
    draw.line((639, 0, 639, 719), fill=(60, 60, 60), width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.jpg")
    canvas.save(
        temporary, format="JPEG", quality=85, subsampling=2, optimize=True
    )
    os.replace(temporary, destination)


def copy_image_without_label_path(source: Path, destination: Path) -> None:
    """Expose unchanged input bytes under an opaque, non-label-bearing path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _baseline_predictions(baseline_dir: Path, seed: int, corruption: str) -> tuple[list[int], list[int]]:
    source_path = baseline_dir / f"Unadapted_seed_{seed}.json"
    after_path = baseline_dir / f"ProtoTTA_final_adaptive_router_seed_{seed}.json"
    source = load_json(source_path)["results"][SOURCE_METHOD][corruption]["5"]
    after = load_json(after_path)["results"][METHOD][corruption]["5"]
    return list(source["online_predictions"]), list(after["online_predictions"])


def export_stream(args: argparse.Namespace) -> Path:
    seed, corruption = args.seed, args.corruption
    output_dir = args.output_dir.resolve() / f"seed_{seed}" / corruption
    public_path = output_dir / "public.json"
    complete_path = output_dir / "export.complete.json"
    if complete_path.is_file() and not args.force:
        return complete_path

    set_random_seed(seed)
    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    dataset_root = args.data_dir / corruption / "5"
    transform = transforms.Compose((
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ))
    dataset = datasets.ImageFolder(str(dataset_root), transform=transform)
    if tuple(dataset.classes) != CLASS_NAMES:
        raise RuntimeError(f"class order drift: {dataset.classes} != {list(CLASS_NAMES)}")
    indices = fixed_stream_indices(len(dataset), "seeded_random", seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, sampler=indices, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )

    source_model = load_model(str(args.model), device)
    source_model.eval().requires_grad_(False)
    adapted_base = load_model(str(args.model), device)
    adapted_base.eval()
    adapted = setup_proto_samplewise_adaptive(
        adapted_base, geo_filter_threshold=0.8, delta0=0.25, top_k=3,
        component_gradient_normalization=True,
        adaptive_controller="absolute_consistency_router",
    )
    source_capture = SpatialDistanceCapture(source_model)
    after_capture = SpatialDistanceCapture(adapted_base)

    expected_before, expected_after = _baseline_predictions(
        args.baseline_dir, seed, corruption
    )
    records: list[dict[str, Any]] = []
    board_metadata: dict[str, Any] = {}
    labels: list[int] = []
    ordered_ids: list[str] = []
    public_ids: list[str] = []
    before_predictions: list[int] = []
    after_predictions: list[int] = []
    offset = 0
    try:
        for batch_number, (images, batch_labels) in enumerate(loader):
            images = images.to(device)
            batch_size = len(images)
            with torch.no_grad():
                before_outputs = source_model(images)
            before_logits = before_outputs[0]
            before_maps = _activation_maps(source_model, source_capture.pop())

            # This is the baseline wrapper call: it returns pre-update logits and
            # performs the usual unsupervised update only for subsequent batches.
            after_outputs = adapted(images)
            after_logits = after_outputs[0]
            after_maps = _activation_maps(adapted_base, after_capture.pop())
            before_probs = before_logits.detach().softmax(dim=1)
            after_probs = after_logits.detach().softmax(dim=1)
            before_pred = before_probs.argmax(dim=1).cpu()
            after_pred = after_probs.argmax(dim=1).cpu()
            labels.extend(int(value) for value in batch_labels.tolist())

            for local_index in range(batch_size):
                stream_index = offset + local_index
                canonical_index = indices[stream_index]
                image_path = Path(dataset.samples[canonical_index][0]).resolve()
                canonical_sample_id = str(
                    image_path.relative_to(dataset_root.resolve())
                )
                ordered_ids.append(canonical_sample_id)
                sample_id = f"sample-{stream_index:05d}"
                public_ids.append(sample_id)
                safe_image_path = (
                    output_dir / "images"
                    / f"{stream_index:05d}{image_path.suffix.lower()}"
                )
                copy_image_without_label_path(image_path, safe_image_path)
                bp = int(before_pred[local_index].item())
                ap = int(after_pred[local_index].item())
                bmsp = float(before_probs[local_index, bp].item())
                amsp = float(after_probs[local_index, ap].item())
                before_predictions.append(bp)
                after_predictions.append(ap)
                record = {
                    "sample_id": sample_id,
                    "stream_index": stream_index,
                    "corruption": corruption,
                    "severity": 5,
                    "seed": seed,
                    # This opaque byte-for-byte copy prevents the ImageFolder
                    # ground-truth class directory from reaching the VLM.
                    "image_path": str(safe_image_path.resolve()),
                    "before_prediction": bp,
                    "before_class": CLASS_NAMES[bp],
                    "before_msp": bmsp,
                    "after_prediction": ap,
                    "after_class": CLASS_NAMES[ap],
                    "after_msp": amsp,
                    "changed": bp != ap,
                    "board_path": None,
                }
                if bp != ap:
                    before_predicted, before_competing = _prototype_rows(
                        source_model, before_maps[local_index], bp
                    )
                    after_predicted, after_competing = _prototype_rows(
                        adapted_base, after_maps[local_index], ap
                    )
                    board_path = output_dir / "boards" / f"{stream_index:05d}.jpg"
                    with Image.open(image_path) as handle:
                        corrupted = handle.convert("RGB")
                    render_board(
                        board_path, corrupted,
                        {"class": CLASS_NAMES[bp], "msp": bmsp,
                         "predicted": before_predicted, "competing": before_competing},
                        {"class": CLASS_NAMES[ap], "msp": amsp,
                         "predicted": after_predicted, "competing": after_competing},
                        source_model, adapted_base, args.prototype_dir,
                    )
                    record["board_path"] = str(board_path.resolve())
                    def serializable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                        return [{key: value for key, value in row.items()
                                 if key != "focus_map"} for row in rows]
                    board_metadata[sample_id] = {
                        "before": {
                            "predicted_class_prototypes": serializable(before_predicted),
                            "competing_class_prototypes": serializable(before_competing),
                        },
                        "after": {
                            "predicted_class_prototypes": serializable(after_predicted),
                            "competing_class_prototypes": serializable(after_competing),
                        },
                        "contribution_definition": (
                            "prototype_activation[p] * "
                            "last_layer_weight[associated_class,p]"
                        ),
                    }
                records.append(record)
            offset += batch_size
            # This partial checkpoint is label-free and safe for monitoring.
            atomic_write_json(output_dir / "export.progress.json", {
                "status": "running", "seed": seed, "corruption": corruption,
                "completed_batches": batch_number + 1, "completed_samples": offset,
                "changed_samples": sum(record["changed"] for record in records),
            })
    finally:
        source_capture.close()
        after_capture.close()

    if before_predictions != expected_before:
        mismatch = next((i for i, pair in enumerate(zip(before_predictions, expected_before))
                         if pair[0] != pair[1]), None)
        raise RuntimeError(f"BEFORE trajectory differs from reported baseline at {mismatch}")
    if after_predictions != expected_after:
        mismatch = next((i for i, pair in enumerate(zip(after_predictions, expected_after))
                         if pair[0] != pair[1]), None)
        raise RuntimeError(f"AFTER trajectory differs from reported ProtoTTA at {mismatch}")
    print(
        "EXPORT_BASELINE_HASHES_VERIFIED "
        f"seed={seed} corruption={corruption} samples={len(records)} "
        f"before={sha256_lines(before_predictions)} "
        f"after={sha256_lines(after_predictions)}",
        flush=True,
    )
    if len(records) != len(dataset) or len(set(ordered_ids)) != len(dataset):
        raise RuntimeError("export is not a complete, unique test-set permutation")

    public = {
        "schema_version": 1,
        "status": "complete",
        "seed": seed,
        "corruption": corruption,
        "severity": 5,
        "records": records,
        "protocol": {
            "batch_size": 64,
            "stream_order": "seeded_random",
            "stream_order_seed": seed,
            "stream_order_algorithm": STREAM_ORDER_ALGORITHM,
            "reset": "fresh source weights at start of this corruption",
            "continuous_within_corruption": True,
            "after_semantics": "pre-update logits; update applies to subsequent batches",
            "prediction_level_rollback_only": True,
            "prototta": {
                "method": METHOD,
                "optimizer": "Adam", "learning_rate": 0.001,
                "adaptation_steps": 1, "adaptation_mode": "batchnorm_addon",
                "geo_filter_threshold": 0.8, "delta0": 0.25, "top_k": 3,
                "component_gradient_normalization": True,
                "adaptive_controller": "absolute_consistency_router",
            },
        },
        "integrity": {
            "num_samples": len(records),
            "num_changed": sum(record["changed"] for record in records),
            "ordered_sample_id_hash_sha256": sha256_lines(ordered_ids),
            "before_prediction_hash_sha256": sha256_lines(before_predictions),
            "after_prediction_hash_sha256": sha256_lines(after_predictions),
            "reported_baselines_matched_exactly": True,
            "model_sha256": sha256_file(args.model),
        },
    }
    # Labels are intentionally separate. VLM scoring refuses to read this file.
    label_payload = {
        "schema_version": 1, "seed": seed, "corruption": corruption,
        "ordered_sample_ids": public_ids, "targets": labels,
        "canonical_ordered_sample_id_hash_sha256": sha256_lines(ordered_ids),
        "target_hash_sha256": sha256_lines(labels),
    }
    atomic_write_json(public_path, public)
    atomic_write_json(output_dir / "board_metadata.json", board_metadata)
    labels_path = output_dir / "sealed_targets.json"
    atomic_write_json(labels_path, label_payload)
    labels_path.chmod(0o600)
    atomic_write_json(complete_path, {
        "status": "complete", "public_path": str(public_path),
        "sealed_targets_path": str(labels_path),
        "num_samples": len(records), "num_changed": public["integrity"]["num_changed"],
    })
    return complete_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corruption", choices=CORRUPTIONS, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prototype-dir", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--gpuid", default="0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = export_stream(args)
    print(json.dumps({"status": "complete", "artifact": str(path)}))


if __name__ == "__main__":
    main()
