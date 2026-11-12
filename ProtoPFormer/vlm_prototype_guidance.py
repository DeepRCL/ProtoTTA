#!/usr/bin/env python3
"""Extract source-only ProtoPFormer prototype evidence images.

For every local prototype, the highest-activating image of the official
Stanford Dogs training split is saved with its activation overlay
(``local_XXXX_source.jpg`` / ``local_XXXX_overlay.jpg``); global prototypes keep
their top source images.  Target (test) images are never used.  The overlays are
the prototype images shown on the judge's reasoning boards
(``export_vlm_consensus_evidence.py --prototype-evidence``).
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from evaluate_robustness_dogs import load_model

ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = ROOT / "output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth"
DEFAULT_SOURCE = ROOT / "datasets/stanford_dogs"
DEFAULT_RESULTS = ROOT / "results/prototype_evidence"


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


def _push(heap: list, score: float, record: dict, keep: int = 2) -> None:
    item = (score, record["path"], record)
    if len(heap) < keep:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {"extract": extract}[arguments.command](arguments)
