#!/usr/bin/env python3
"""Create and score source-only ProtoPFormer prototype evidence boards."""
from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
PV_ROOT = ROOT.parent / "ProtoViT"
sys.path.insert(0, str(PV_ROOT))
import vlm_eval  # noqa: E402

from evaluate_robustness_dogs import load_model  # noqa: E402


DEFAULT_MODEL = ROOT / "output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth"
DEFAULT_SOURCE = ROOT / "datasets/stanford_dogs"
DEFAULT_RESULTS = ROOT / "results/vlm_purified"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class DogsSourceTrain(Dataset):
    """Official source training split; target test images are never used."""

    def __init__(self, root: Path):
        split = scipy.io.loadmat(root / "train_list.mat")
        self.paths = [root / "Images" / str(item[0]) for item in split["file_list"].squeeze()]
        self.labels = split["labels"].squeeze().astype(np.int64) - 1
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), int(self.labels[index]), str(self.paths[index])


def class_names(source: Path) -> list[str]:
    return [path.name.split("-", 1)[-1].replace("_", " ")
            for path in sorted((source / "Images").iterdir()) if path.is_dir()]


def _push(heap: list, score: float, record: dict, keep: int = 2) -> None:
    item = (score, record["path"], record)
    if len(heap) < keep:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _evidence_images(record: dict, source_out: Path, overlay_out: Path) -> None:
    image = Image.open(record["path"]).convert("RGB").resize((224, 224))
    image.save(source_out, quality=92)
    activation = np.asarray(record["activation"], dtype=np.float32).reshape(-1)
    attention = np.asarray(record["attention"], dtype=np.float32).reshape(-1)
    dense = np.zeros(attention.size, dtype=np.float32)
    selected = np.sort(np.argpartition(attention, -activation.size)[-activation.size:])
    dense[selected] = activation
    side = int(round(np.sqrt(dense.size)))
    dense = dense.reshape(side, side)
    dense -= dense.min()
    dense /= max(float(dense.max()), 1e-8)
    heat = Image.fromarray(np.uint8(255 * dense)).resize((224, 224), Image.Resampling.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32) / 255.0
    base = np.asarray(image, dtype=np.float32)
    colored = np.zeros_like(base)
    colored[..., 0] = 255.0 * heat
    colored[..., 1] = 80.0 * heat
    overlay = np.uint8(np.clip(0.72 * base + 0.28 * colored, 0, 255))
    Image.fromarray(overlay).save(overlay_out, quality=92)


def extract(args) -> None:
    device = torch.device("cuda")
    model = load_model(args.model, device).eval()
    dataset = DogsSourceTrain(args.source)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    local = [[] for _ in range(model.num_prototypes)]
    global_ = [[] for _ in range(model.num_prototypes_global)]
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            _, aux = model(images)
            attention, distances = aux[0], aux[1]
            local_scores, global_scores = aux[4], aux[5]
            maps = model.distance_2_similarity(distances)
            for row, label in enumerate(labels.tolist()):
                for offset in range(10):
                    index = label * 10 + offset
                    _push(local[index], float(local_scores[row, index]), {
                        "path": paths[row],
                        "attention": attention[row].detach().float().cpu().reshape(-1).tolist(),
                        "activation": maps[row, index].detach().float().cpu().reshape(-1).tolist(),
                    })
                for offset in range(5):
                    index = label * 5 + offset
                    _push(global_[index], float(global_scores[row, index]), {"path": paths[row]})
    evidence = args.results / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "split": "official Stanford Dogs source train split",
                "num_source_images": len(dataset), "local": {}, "global": {}}
    for index, heap in enumerate(local):
        records = [item[2] for item in sorted(heap, reverse=True)]
        _evidence_images(records[0], evidence / f"local_{index:04d}_source.jpg",
                         evidence / f"local_{index:04d}_overlay.jpg")
        manifest["local"][str(index)] = [{"path": x["path"]} for x in records]
    for index, heap in enumerate(global_):
        records = [item[2] for item in sorted(heap, reverse=True)]
        for rank, record in enumerate(records):
            Image.open(record["path"]).convert("RGB").resize((224, 224)).save(
                evidence / f"global_{index:04d}_source{rank}.jpg", quality=92)
        manifest["global"][str(index)] = records
    write_json(args.results / "evidence_manifest.json", manifest)


def render(args) -> None:
    names = class_names(args.source)
    board_dir = args.results / "boards"
    board_dir.mkdir(parents=True, exist_ok=True)
    evidence = args.results / "evidence"
    for class_index, name in enumerate(names):
        canvas = Image.new("RGB", (1900, 920), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), f"Dog class {class_index}: {name}", fill="black", font=_font(30))
        for local in range(10):
            col, row = local % 5, local // 5
            x, y = 15 + col * 376, 60 + row * 285
            draw.rectangle((x, y, x + 365, y + 275), outline=(80, 80, 80), width=2)
            draw.text((x + 5, y + 4), f"Local L{local}", fill="black", font=_font(18))
            index = class_index * 10 + local
            for side, suffix in enumerate(("source", "overlay")):
                image = Image.open(evidence / f"local_{index:04d}_{suffix}.jpg")
                image.thumbnail((170, 220))
                canvas.paste(image, (x + 5 + side * 178, y + 30))
        for glob in range(5):
            x, y = 15 + glob * 376, 630
            draw.rectangle((x, y, x + 365, y + 275), outline=(80, 80, 80), width=2)
            draw.text((x + 5, y + 4), f"Global G{glob}", fill="black", font=_font(18))
            index = class_index * 5 + glob
            for side in range(2):
                image = Image.open(evidence / f"global_{index:04d}_source{side}.jpg")
                image.thumbnail((170, 220))
                canvas.paste(image, (x + 5 + side * 178, y + 30))
        canvas.save(board_dir / f"class_{class_index:03d}.jpg", quality=92)


def _ranking(value, count: int) -> list[int]:
    ranking = []
    for item in value if isinstance(value, list) else []:
        # Qwen may return the requested visual labels ("L4"/"G2") rather
        # than bare integers.  They carry the same unambiguous ranking.
        match = re.search(r"-?\d+", str(item))
        if match is None:
            continue
        item = int(match.group())
        if 0 <= item < count and item not in ranking:
            ranking.append(item)
    ranking.extend(item for item in range(count) if item not in ranking)
    if len(ranking) != count:
        raise ValueError("Invalid ranking")
    return ranking


def rank(args) -> None:
    names = class_names(args.source)
    scorer = vlm_eval.VLMScorer(args.model_id, args.max_new_tokens, enable_thinking=False)
    for class_index, name in enumerate(names):
        if class_index % args.num_shards != args.shard_index:
            continue
        output = args.results / "rankings" / f"class_{class_index:03d}.json"
        if output.exists():
            continue
        prompt = f"""Audit the learned local prototype bank for dog class '{name}'.
The first two rows show ten LOCAL prototypes L0--L9: source image and its activation overlay. Rank them from best to worst. Prefer localized, recognizable, breed-compatible dog parts; penalize background, borders, and artifacts.
Ignore the global-prototype reference row: global representations do not provide spatially distinct evidence and are deliberately left unchanged in this experiment.
Make a strict local ranking and use each index once. Return exactly one JSON object:
{{"local_ranking":[10 local indices],"rationale":"one short sentence"}}"""
        parsed, raw, _ = scorer.generate_json(
            [args.results / "boards" / f"class_{class_index:03d}.jpg"], prompt)
        write_json(output, {
            "schema_version": 1, "model_id": args.model_id, "class_index": class_index,
            "class_name": name, "local_ranking": _ranking(parsed.get("local_ranking"), 10),
            "global_ranking": _ranking(parsed.get("global_ranking"), 5),
            "rationale": parsed.get("rationale", ""), "raw_response": raw,
        })


def aggregate(args) -> None:
    names = class_names(args.source)
    local_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.25, 0.10, 0.05]
    # The global branch has no spatially distinct evidence: its top examples
    # were frequently identical across all five prototypes.  Keep it neutral
    # instead of turning forced tie-breaking into supervision.
    global_weights = [1.0] * 5
    local_quality = np.zeros(len(names) * 10)
    global_quality = np.zeros(len(names) * 5)
    for class_index in range(len(names)):
        result = json.loads((args.results / "rankings" / f"class_{class_index:03d}.json").read_text())
        for rank_index, local in enumerate(_ranking(result["local_ranking"], 10)):
            local_quality[class_index * 10 + local] = local_weights[rank_index]
        for rank_index, glob in enumerate(_ranking(result.get("global_ranking"), 5)):
            global_quality[class_index * 5 + glob] = global_weights[rank_index]
    write_json(args.results / "prototype_quality_ranked.json", {
        "schema_version": 1, "model_id": "Qwen/Qwen3.6-35B-A3B",
        "source_data": "official source-training split only; no target images or labels",
        "scope": "VLM ranks spatially interpretable local prototypes; global branch remains neutral",
        "local_rank_weights": local_weights, "global_rank_weights": global_weights,
        "local_quality": local_quality.tolist(), "global_quality": global_quality.tolist(),
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "render", "rank", "aggregate"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {"extract": extract, "render": render, "rank": rank, "aggregate": aggregate}[arguments.command](arguments)
