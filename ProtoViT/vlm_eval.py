#!/usr/bin/env python3
"""VLM-based reasoning evaluation for ProtoViT TTA methods."""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

from preprocess import mean, std, undo_preprocess_input_function
from settings import img_size


LOGGER = logging.getLogger("vlm_eval")
SCRIPT_DIR = Path(__file__).resolve().parent
eata_adapt = None
inference_utils = None
viz = None
_MODEL_MODULE = None
_PUSH_GREEDY_MODULE = None

CORRUPTION_TYPES = [
    "gaussian_noise",
    "fog",
    "gaussian_blur",
    "elastic_transform",
    "brightness",
    "jpeg_compression",
    "contrast",
    "defocus_blur",
    "frost",
    "impulse_noise",
    "pixelate",
    "shot_noise",
    "speckle_noise",
]

METHOD_ORDER = ["unadapted", "tent", "eata", "sar", "memo", "prototta"]
TABLE_METHOD_ORDER = ["unadapted", "memo", "sar", "tent", "eata", "prototta"]
BATCH_PRECOMPUTE_METHODS = {"unadapted", "prototta"}
DISPLAY_NAMES = {
    "unadapted": "Unadapted",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "memo": "Memo",
    "prototta": "ProtoTTA",
}

PROTO_TTA_DEFAULTS = {
    "use_importance": True,
    "use_confidence": True,
    "reset_mode": None,
    "reset_frequency": 10,
    "confidence_threshold": 0.7,
    "ema_alpha": 0.999,
    "use_geometric_filter": True,
    "geo_filter_threshold": 0.92,
    "consensus_strategy": "top_k_mean",
    "consensus_ratio": 0.5,
    "adaptation_mode": "layernorm_attn_bias",
    "use_ensemble_entropy": False,
}

PROTOTYPE_PATTERNS = [
    "prototype-imgbbox-original{idx}.png",
    "prototype-img_vis_{idx}.png",
    "prototype-img-original{idx}.png",
    "prototype-img{idx}.png",
    "prototype{idx}.png",
]


def lazy_import_runtime_modules() -> None:
    global eata_adapt
    global inference_utils
    global viz
    global _MODEL_MODULE
    global _PUSH_GREEDY_MODULE

    if inference_utils is not None and viz is not None and eata_adapt is not None:
        return

    import eata_adapt as eata_module
    import interpretability_viz as viz_module
    import model as model_module  # noqa: F401
    import push_greedy as push_greedy_module  # noqa: F401
    import run_inference as inference_module

    eata_adapt = eata_module
    viz = viz_module
    inference_utils = inference_module
    _MODEL_MODULE = model_module
    _PUSH_GREEDY_MODULE = push_greedy_module


def install_timm_checkpoint_compat() -> None:
    """Map old timm pickle paths to current ones before torch.load.

    Older ProtoViT checkpoints were serialized against timm versions where
    many helpers lived under `timm.models.layers.*`. Newer timm versions moved
    those modules to `timm.layers.*`. Torch deserialization resolves classes by
    their original module path, so we pre-register aliases for the whole old
    namespace instead of patching one missing module at a time.
    """
    try:
        import importlib
        import pkgutil
        import timm.layers as current_layers

        sys.modules["timm.models.layers"] = current_layers

        if getattr(current_layers, "__path__", None):
            for module_info in pkgutil.iter_modules(current_layers.__path__):
                old_name = f"timm.models.layers.{module_info.name}"
                new_name = f"timm.layers.{module_info.name}"
                try:
                    sys.modules[old_name] = importlib.import_module(new_name)
                except Exception:
                    continue
    except Exception:
        pass

    # Monkey-patch PatchEmbed to handle missing attributes from old checkpoints
    try:
        import torch.nn as nn
        import timm.layers.patch_embed
        
        def _patch_embed_setstate(self, state):
            nn.Module.__setstate__(self, state)
            if not hasattr(self, 'strict_img_size'):
                self.strict_img_size = False
            if not hasattr(self, 'dynamic_img_pad'):
                self.dynamic_img_pad = False
                
        timm.layers.patch_embed.PatchEmbed.__setstate__ = _patch_embed_setstate
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM quality scoring and adaptation narratives for ProtoViT."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR / "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth",
        help="Path to the saved ProtoViT model.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=SCRIPT_DIR / "datasets/cub200_cropped/test_cropped",
        help="Path to the clean CUB-200-C test set.",
    )
    parser.add_argument(
        "--corrupted-dir",
        type=Path,
        default=SCRIPT_DIR / "datasets/cub200_c",
        help="Path to the pre-generated corrupted CUB-200-C dataset.",
    )
    parser.add_argument(
        "--prototype-dir",
        type=Path,
        default=None,
        help="Optional override for the prototype image directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "vlm_eval",
        help="Base directory for all VLM evaluation artifacts.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=5,
        help="Corruption severity to evaluate.",
    )
    parser.add_argument(
        "--part",
        choices=["A", "B"],
        default=None,
        help="Run only Part A or Part B. Omit to run both sequentially.",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_ORDER,
        default=None,
        help="Run a single Part A method for testing or resuming.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for subset creation and Fisher estimation.",
    )
    parser.add_argument(
        "--gpuid",
        type=str,
        default="0",
        help="GPU id to expose via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--memo-lr",
        type=float,
        default=2.5e-4,
        help="Learning rate for MEMO.",
    )
    parser.add_argument(
        "--memo-batch-size",
        type=int,
        default=64,
        help="Number of augmented views per MEMO step.",
    )
    parser.add_argument(
        "--memo-steps",
        type=int,
        default=1,
        help="Number of MEMO adaptation steps per sample.",
    )
    parser.add_argument(
        "--tta-steps",
        type=int,
        default=1,
        help="Number of test-time adaptation steps for Tent/EATA/SAR/ProtoTTA.",
    )
    parser.add_argument(
        "--eata-fisher-source",
        choices=["subset", "clean"],
        default="subset",
        help="Use the selected corrupted subset or clean data to estimate EATA Fishers.",
    )
    parser.add_argument(
        "--fisher-samples",
        type=int,
        default=500,
        help="Maximum number of samples for EATA Fisher estimation.",
    )
    parser.add_argument(
        "--vlm-model-id",
        type=str,
        default="Qwen/Qwen3-VL-32B-Thinking",
        help="Hugging Face model id for the VLM scorer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2056,
        help="Maximum tokens to generate per VLM call.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of subset samples to process, useful for 1-sample VLM tests.",
    )
    parser.add_argument(
        "--rebuild-subset",
        action="store_true",
        help="Rebuild subset.json from the saved full-dataset unadapted precompute cache.",
    )
    parser.add_argument(
        "--precompute-save-every",
        type=int,
        default=10,
        help="Save full-dataset precompute cache every N batches instead of every batch.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def humanize_class_name(raw_name: str) -> str:
    cleaned = raw_name.split(".", 1)[1] if "." in raw_name else raw_name
    return cleaned.replace("_", " ")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    restored = undo_preprocess_input_function(image_tensor.unsqueeze(0).detach().cpu())[0]
    restored = restored.numpy().transpose(1, 2, 0)
    return np.clip(restored, 0, 1)


def save_rgb_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.imsave(output_path, tensor_to_rgb(image_tensor))


def resolve_prototype_dir(model_path: Path, explicit_dir: Optional[Path]) -> Path:
    if explicit_dir is not None:
        return resolve_path(explicit_dir)
    model_dir = model_path.resolve().parent
    candidates = [
        model_dir / "img",
        model_dir / "prototype_imgs",
        model_dir / "prototype-img",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a prototype directory next to {model_path}. "
        f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def load_corruption_dataset(
    corrupted_dir: Path, corruption_type: str, severity: int
) -> datasets.ImageFolder:
    dataset_root = corrupted_dir / corruption_type / str(severity)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing corruption directory: {dataset_root}")
    return datasets.ImageFolder(str(dataset_root), transform=build_transform())


def load_clean_dataset(clean_dir: Path) -> datasets.ImageFolder:
    if not clean_dir.exists():
        raise FileNotFoundError(f"Missing clean directory: {clean_dir}")
    return datasets.ImageFolder(str(clean_dir), transform=build_transform())


def sample_stem(sample: Dict) -> str:
    return f"sample_{sample['sample_idx']}_{sample['corruption_type']}"


def sample_corrupted_image_path(sample: Dict, corrupted_dir: Path, severity: int) -> Path:
    return corrupted_dir / sample["corruption_type"] / str(severity) / sample["image_path"]


def sample_clean_image_path(sample: Dict, clean_dir: Path) -> Path:
    return clean_dir / sample["image_path"]


def precompute_dir(results_dir: Path, method: str) -> Path:
    return results_dir / "precompute" / method


def precompute_file(results_dir: Path, method: str, corruption_type: str) -> Path:
    return precompute_dir(results_dir, method) / f"{corruption_type}.json"


def precompute_key(corruption_type: str, image_path: str) -> str:
    return f"{corruption_type}::{image_path}"


def load_precompute_map(results_dir: Path, method: str, corruption_type: str) -> Dict[str, Dict]:
    path = precompute_file(results_dir, method, corruption_type)
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload.get("entries", {})


def save_precompute_map(
    results_dir: Path,
    method: str,
    corruption_type: str,
    severity: int,
    entries: Dict[str, Dict],
) -> None:
    write_json(
        precompute_file(results_dir, method, corruption_type),
        {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "corruption_type": corruption_type,
            "severity": severity,
            "num_entries": len(entries),
            "entries": entries,
        },
    )


def load_precomputed_entry(results_dir: Path, method: str, sample: Dict) -> Optional[Dict]:
    entries = load_precompute_map(results_dir, method, sample["corruption_type"])
    return entries.get(precompute_key(sample["corruption_type"], sample["image_path"]))


def precompute_complete(results_dir: Path, method: str, severity: int) -> bool:
    for corruption_type in CORRUPTION_TYPES:
        path = precompute_file(results_dir, method, corruption_type)
        if not path.exists():
            return False
        payload = load_json(path)
        if int(payload.get("severity", -1)) != severity:
            return False
        if payload.get("num_entries", 0) <= 0:
            return False
    return True


def predicted_quota(total_quota: int) -> Tuple[int, int]:
    if total_quota >= 8:
        return 3, 5
    easy_quota = max(1, total_quota - 3)
    hard_quota = total_quota - easy_quota
    return easy_quota, hard_quota


def build_subset_manifest(
    subset_path: Path,
    results_dir: Path,
    severity: int,
    rebuild: bool = False,
) -> Dict:
    if subset_path.exists() and not rebuild:
        manifest = load_json(subset_path)
        if len(manifest.get("samples", [])) != 100:
            raise ValueError(f"Existing subset manifest is invalid: {subset_path}")
        LOGGER.info("Using existing subset manifest at %s", subset_path)
        return manifest

    LOGGER.info("Creating fixed 100-sample subset at %s", subset_path)
    rng = random.Random(42)

    subset_samples: List[Dict] = []
    quotas = [8] * len(CORRUPTION_TYPES)
    quotas[-1] = 4

    for corruption_type, quota in zip(CORRUPTION_TYPES, quotas):
        payload = load_json(precompute_file(results_dir, "unadapted", corruption_type))
        entries = list(payload.get("entries", {}).values())
        if not entries:
            raise RuntimeError(
                f"Missing unadapted precompute entries for {corruption_type}. "
                "Run the batch precompute step first."
            )

        easy_candidates = [entry for entry in entries if entry["is_correct"]]
        hard_candidates = [entry for entry in entries if not entry["is_correct"]]
        easy_quota, hard_quota = predicted_quota(quota)

        selected_easy = rng.sample(easy_candidates, min(easy_quota, len(easy_candidates)))
        selected_hard = rng.sample(hard_candidates, min(hard_quota, len(hard_candidates)))

        selected_keys = {
            (item["sample_idx"], item["image_path"]) for item in selected_easy + selected_hard
        }
        remaining_candidates = [
            entry
            for entry in entries
            if (entry["sample_idx"], entry["image_path"]) not in selected_keys
        ]
        remaining_needed = quota - len(selected_easy) - len(selected_hard)
        filler = rng.sample(remaining_candidates, remaining_needed)

        chosen = sorted(
            selected_easy + selected_hard + filler,
            key=lambda item: item["sample_idx"],
        )
        for item in chosen:
            subset_samples.append(
                {
                    "sample_idx": int(item["sample_idx"]),
                    "corruption_type": item["corruption_type"],
                    "ground_truth_class": item["ground_truth_class"],
                    "ground_truth_index": int(item["ground_truth_index"]),
                    "unadapted_correct": bool(item["is_correct"]),
                    "image_path": item["image_path"],
                    "class_folder": item.get("class_folder"),
                }
            )

    if len(subset_samples) != 100:
        raise RuntimeError(f"Subset creation failed: expected 100 samples, got {len(subset_samples)}")

    for subset_position, sample in enumerate(subset_samples):
        sample["subset_position"] = subset_position

    manifest = {
        "seed": 42,
        "severity": severity,
        "corruption_types": CORRUPTION_TYPES,
        "samples": subset_samples,
    }
    write_json(subset_path, manifest)
    LOGGER.info("Saved fixed subset manifest with %d samples", len(subset_samples))
    return manifest


def build_subset_clean_loader(
    clean_dir: Path, batch_size: int, fisher_samples: int
) -> torch.utils.data.DataLoader:
    dataset = load_clean_dataset(clean_dir)
    if len(dataset) > fisher_samples:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[:fisher_samples]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


class ManifestSubsetDataset(torch.utils.data.Dataset):
    """A deterministic dataset wrapper over the saved subset manifest."""

    def __init__(self, manifest_samples: Sequence[Dict], corrupted_dir: Path, severity: int):
        self.samples = list(manifest_samples)
        self.corrupted_dir = corrupted_dir
        self.severity = severity
        self.transform = build_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image_path = sample_corrupted_image_path(sample, self.corrupted_dir, self.severity)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(sample["ground_truth_index"])


def build_subset_corrupted_loader(
    manifest_samples: Sequence[Dict],
    corrupted_dir: Path,
    severity: int,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    dataset = ManifestSubsetDataset(manifest_samples, corrupted_dir, severity)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def compute_eata_fishers(
    model_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    subset_samples: Sequence[Dict],
) -> Dict:
    LOGGER.info("Computing EATA Fishers from %s data", args.eata_fisher_source)
    fisher_model = torch.load(str(model_path), weights_only=False)
    fisher_model = fisher_model.to(device)
    fisher_model = eata_adapt.configure_model(fisher_model)

    if args.eata_fisher_source == "clean":
        loader = build_subset_clean_loader(args.clean_dir, args.batch_size, args.fisher_samples)
    else:
        loader = build_subset_corrupted_loader(
            subset_samples, args.corrupted_dir, args.severity, args.batch_size
        )

    fishers = eata_adapt.compute_fishers(
        fisher_model, loader, device, num_samples=args.fisher_samples
    )
    del fisher_model
    torch.cuda.empty_cache()
    return fishers


def setup_method_model(
    method: str,
    model_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    subset_samples: Sequence[Dict],
    eata_fishers: Optional[Dict],
):
    base_model = torch.load(str(model_path), weights_only=False)
    base_model = base_model.to(device)

    inference_utils.cfg.OPTIM.STEPS = args.tta_steps
    inference_utils.cfg.MODEL.EPISODIC = False

    if method == "unadapted":
        base_model.eval()
        return base_model
    if method == "tent":
        return inference_utils.setup_tent(base_model)
    if method == "eata":
        fishers = eata_fishers or compute_eata_fishers(model_path, device, args, subset_samples)
        return inference_utils.setup_eata(base_model, fishers)
    if method == "sar":
        return inference_utils.setup_sar(base_model)
    if method == "memo":
        return inference_utils.setup_memo(
            base_model,
            lr=args.memo_lr,
            batch_size=args.memo_batch_size,
            steps=args.memo_steps,
        )
    if method == "prototta":
        return inference_utils.setup_proto_entropy(
            base_model,
            use_importance=PROTO_TTA_DEFAULTS["use_importance"],
            use_confidence=PROTO_TTA_DEFAULTS["use_confidence"],
            reset_mode=PROTO_TTA_DEFAULTS["reset_mode"],
            reset_frequency=PROTO_TTA_DEFAULTS["reset_frequency"],
            confidence_threshold=PROTO_TTA_DEFAULTS["confidence_threshold"],
            ema_alpha=PROTO_TTA_DEFAULTS["ema_alpha"],
            use_geometric_filter=PROTO_TTA_DEFAULTS["use_geometric_filter"],
            geo_filter_threshold=PROTO_TTA_DEFAULTS["geo_filter_threshold"],
            consensus_strategy=PROTO_TTA_DEFAULTS["consensus_strategy"],
            consensus_ratio=PROTO_TTA_DEFAULTS["consensus_ratio"],
            adaptation_mode=PROTO_TTA_DEFAULTS["adaptation_mode"],
            use_ensemble_entropy=PROTO_TTA_DEFAULTS["use_ensemble_entropy"],
        )
    raise ValueError(f"Unsupported method: {method}")


def normalize_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 3:
        return outputs
    raise TypeError("Expected model outputs to be a (logits, min_distances, values) tuple.")


def get_underlying_model(eval_model):
    return eval_model.model if hasattr(eval_model, "model") else eval_model


def clone_proto(proto: Dict) -> Dict:
    cloned = dict(proto)
    cloned["slots"] = np.asarray(proto["slots"]).copy()
    cloned["patch_locations"] = tuple(
        np.asarray(item).copy() for item in proto["patch_locations"]
    )
    return cloned


def proto_class_index(proto: Dict) -> int:
    if "class" in proto:
        return int(proto["class"])
    return int(proto["class_index"])


def proto_class_name(proto: Dict, class_names: Sequence[str]) -> str:
    if "class_name" in proto:
        return str(proto["class_name"])
    return humanize_class_name(class_names[proto_class_index(proto)])


def select_predicted_class_prototypes(
    all_proto_results: Sequence[Dict],
    pred_class: int,
    ppnet,
    top_k: int = 5,
) -> List[Dict]:
    weights = ppnet.last_layer.weight[pred_class, :].detach().cpu().numpy()
    filtered = [
        clone_proto(proto)
        for proto in all_proto_results
        if proto["class"] == pred_class and weights[proto["proto_idx"]] > 0
    ]

    if filtered:
        for proto in filtered:
            proto["contribution"] = proto["activation"] * weights[proto["proto_idx"]]
        filtered.sort(key=lambda item: item["contribution"], reverse=True)
        return filtered[:top_k]

    positive_weight_pool = [
        clone_proto(proto)
        for proto in all_proto_results
        if weights[proto["proto_idx"]] > 0
    ]
    if positive_weight_pool:
        for proto in positive_weight_pool:
            proto["contribution"] = proto["activation"] * weights[proto["proto_idx"]]
        positive_weight_pool.sort(key=lambda item: item["contribution"], reverse=True)
        return positive_weight_pool[:top_k]

    fallback = [clone_proto(proto) for proto in all_proto_results[:top_k]]
    for proto in fallback:
        proto["contribution"] = proto["activation"] * max(0.0, proto["connection_weight"])
    return fallback


def select_any_class_prototypes(
    all_proto_results: Sequence[Dict], ppnet, top_k: int = 10
) -> List[Dict]:
    all_weights = ppnet.last_layer.weight.detach().cpu().numpy()
    max_weights = np.maximum(all_weights, 0.0).max(axis=0)
    ranked = [clone_proto(proto) for proto in all_proto_results]
    for proto in ranked:
        proto["contribution"] = proto["activation"] * max_weights[proto["proto_idx"]]
    positive = [proto for proto in ranked if proto["contribution"] > 0]
    positive.sort(key=lambda item: item["contribution"], reverse=True)
    return (positive or ranked)[:top_k]


def build_precomputed_sample_meta(
    method: str,
    sample_idx: int,
    corruption_type: str,
    image_path: str,
    gt_idx: int,
    gt_class: str,
    pred_idx: int,
    class_names: Sequence[str],
    predicted_top: Sequence[Dict],
    any_class_top: Sequence[Dict],
) -> Dict:
    return {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "sample_idx": int(sample_idx),
        "corruption_type": corruption_type,
        "image_path": image_path,
        "ground_truth_class": gt_class,
        "ground_truth_index": int(gt_idx),
        "predicted_class": humanize_class_name(class_names[pred_idx]),
        "predicted_index": int(pred_idx),
        "is_correct": bool(pred_idx == gt_idx),
        "predicted_top_prototypes": [proto_to_json(proto, class_names) for proto in predicted_top],
        "any_class_top_prototypes": [proto_to_json(proto, class_names) for proto in any_class_top],
    }


def run_full_dataset_precompute(
    method: str,
    args: argparse.Namespace,
    device: torch.device,
    prototype_img_dir: Path,
    class_names: Sequence[str],
    eata_fishers: Optional[Dict],
) -> None:
    if method not in BATCH_PRECOMPUTE_METHODS:
        return
    if precompute_complete(args.results_dir, method, args.severity):
        LOGGER.info("Skipping full-dataset precompute for %s because caches already exist", DISPLAY_NAMES[method])
        return

    LOGGER.info("Starting full-dataset batched precompute for %s", DISPLAY_NAMES[method])

    for corruption_type in CORRUPTION_TYPES:
        dataset = load_corruption_dataset(args.corrupted_dir, corruption_type, args.severity)
        cache = load_precompute_map(args.results_dir, method, corruption_type)
        if len(cache) == len(dataset):
            LOGGER.info(
                "Precompute cache already complete for %s / %s (%d samples)",
                DISPLAY_NAMES[method],
                corruption_type,
                len(cache),
            )
            continue

        eval_model = setup_method_model(method, args.model, device, args, [], eata_fishers)
        ppnet = get_underlying_model(eval_model)
        ppnet.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        cursor = 0
        dirty_batches = 0
        pbar = tqdm(loader, desc=f"precompute:{DISPLAY_NAMES[method]}:{corruption_type}", leave=False)
        for batch_id, (images, labels) in enumerate(pbar, start=1):
            batch_start = time.perf_counter()
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = eval_model(images)
            forward_time = time.perf_counter() - batch_start
            logits, min_distances, values = normalize_outputs(outputs)
            preds = logits.argmax(dim=1)
            proto_start = time.perf_counter()
            batch_proto_results = viz.get_top_k_prototypes_batch(
                ppnet,
                images,
                k=None,
                precomputed_outputs=(logits, min_distances, values),
                sort_by="activation",
            )
            proto_time = time.perf_counter() - proto_start

            for batch_index in range(labels.size(0)):
                dataset_index = cursor + batch_index
                sample_path, _ = dataset.samples[dataset_index]
                rel_path = os.path.relpath(sample_path, dataset.root)
                key = precompute_key(corruption_type, rel_path)
                if key in cache:
                    continue

                label_idx = int(labels[batch_index].item())
                pred_idx = int(preds[batch_index].item())
                all_proto_results, _ = batch_proto_results[batch_index]
                predicted_top = select_predicted_class_prototypes(all_proto_results, pred_idx, ppnet, top_k=5)
                any_class_top = select_any_class_prototypes(all_proto_results, ppnet, top_k=10)
                cache[key] = build_precomputed_sample_meta(
                    method=method,
                    sample_idx=dataset_index,
                    corruption_type=corruption_type,
                    image_path=rel_path,
                    gt_idx=label_idx,
                    gt_class=humanize_class_name(dataset.classes[label_idx]),
                    pred_idx=pred_idx,
                    class_names=class_names,
                    predicted_top=predicted_top,
                    any_class_top=any_class_top,
                )
            cursor += labels.size(0)
            dirty_batches += 1
            if dirty_batches >= args.precompute_save_every or cursor >= len(dataset):
                save_start = time.perf_counter()
                save_precompute_map(args.results_dir, method, corruption_type, args.severity, cache)
                save_time = time.perf_counter() - save_start
                LOGGER.info(
                    "Precompute %s/%s batch %d: forward=%.2fs proto=%.2fs save=%.2fs cache=%d/%d",
                    DISPLAY_NAMES[method],
                    corruption_type,
                    batch_id,
                    forward_time,
                    proto_time,
                    save_time,
                    len(cache),
                    len(dataset),
                )
                dirty_batches = 0

        del eval_model
        torch.cuda.empty_cache()


def prototype_dirs(prototype_img_dir: Path) -> List[Path]:
    return [
        prototype_img_dir,
        prototype_img_dir / "epoch-4",
        prototype_img_dir.parent,
    ]


def load_prototype_patch(prototype_img_dir: Path, proto_idx: int) -> Optional[np.ndarray]:
    for base_dir in prototype_dirs(prototype_img_dir):
        if not base_dir.exists():
            continue
        for pattern in PROTOTYPE_PATTERNS:
            candidate = base_dir / pattern.format(idx=proto_idx)
            if candidate.exists():
                try:
                    return plt.imread(candidate)
                except Exception:
                    continue
    return None


def create_heatmap(proto: Dict, image_size: int = 224) -> np.ndarray:
    heatmap = np.zeros((14, 14), dtype=np.float32)
    score = float(proto.get("contribution", proto["activation"]))
    for slot_index, slot_value in enumerate(proto["slots"]):
        if slot_value > 0:
            h_idx = int(proto["patch_locations"][0][slot_index])
            w_idx = int(proto["patch_locations"][1][slot_index])
            heatmap[h_idx, w_idx] = max(heatmap[h_idx, w_idx], score)
    zoom_factor = image_size / heatmap.shape[0]
    return zoom(heatmap, zoom_factor, order=1)


def prototype_title(proto: Dict, class_names: Sequence[str], rank: int) -> str:
    proto_class = proto_class_name(proto, class_names)
    wrapped = textwrap.fill(proto_class, width=16)
    return (
        f"C{rank} P{proto['proto_idx']}\n"
        f"{wrapped}\n"
        f"{proto.get('contribution', 0.0):.2f}"
    )


def render_reasoning_overview(
    image_tensor: torch.Tensor,
    proto_results: Sequence[Dict],
    title: str,
    output_path: Path,
    prototype_img_dir: Path,
    grid_rows: int,
    grid_cols: int,
    class_names: Sequence[str],
    board_label: str,
) -> None:
    ensure_dir(output_path.parent)
    img_rgb = tensor_to_rgb(image_tensor)
    fig = plt.figure(figsize=(7 + 2.5 * grid_cols, 5.7 + 1.25 * grid_rows))
    content = fig.add_gridspec(
        grid_rows,
        grid_cols + 2,
        width_ratios=[1.15, 1.15] + [1.0] * grid_cols,
        wspace=0.08,
        hspace=0.2,
    )

    raw_ax = fig.add_subplot(content[:, 0])
    raw_ax.imshow(img_rgb)
    raw_ax.set_title("A Raw", fontsize=11, fontweight="bold")
    raw_ax.axis("off")

    overlay_ax = fig.add_subplot(content[:, 1])
    overlay_ax.imshow(img_rgb)
    if proto_results:
        overlay_ax.imshow(
            create_heatmap(proto_results[0], image_size=img_rgb.shape[0]),
            alpha=0.58,
            cmap="hot",
        )
        top_class = proto_class_name(proto_results[0], class_names)
        overlay_ax.set_title(
            "B Focus\n"
            f"P{proto_results[0]['proto_idx']} | {textwrap.shorten(top_class, width=18, placeholder='...')}\n"
            f"{proto_results[0].get('contribution', 0.0):.2f}",
            fontsize=10,
            fontweight="bold",
        )
    else:
        overlay_ax.set_title("B Focus\nnone", fontsize=10, fontweight="bold")
    overlay_ax.axis("off")

    for index in range(grid_rows * grid_cols):
        row = index // grid_cols
        col = index % grid_cols
        ax = fig.add_subplot(content[row, col + 2])
        if index < len(proto_results):
            proto = proto_results[index]
            patch = load_prototype_patch(prototype_img_dir, proto["proto_idx"])
            if patch is not None:
                ax.imshow(patch)
            else:
                ax.text(0.5, 0.5, f"Proto {proto['proto_idx']}\nnot found", ha="center", va="center")
            ax.set_title(prototype_title(proto, class_names, index + 1), fontsize=8.2)
        ax.axis("off")

    fig.suptitle(f"{title}\n{board_label}", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def bar_color_for_proto(proto_class_idx: int, gt_idx: int) -> str:
    return "#2e8b57" if proto_class_idx == gt_idx else "#b22222"


def plot_prototype_bars(
    ax,
    proto_results: Sequence[Dict],
    gt_idx: int,
    delta_mode: bool = False,
    title: str = "",
) -> None:
    if not proto_results:
        ax.text(0.5, 0.5, "No positive prototype contributions", ha="center", va="center")
        ax.set_axis_off()
        return

    labels = [f"P{proto['proto_idx']}" for proto in proto_results]
    values = [float(proto.get("delta_contribution", proto.get("contribution", 0.0))) for proto in proto_results]
    colors = []
    for proto in proto_results:
        if delta_mode:
            colors.append("#228b22" if proto.get("delta_contribution", 0.0) >= 0 else "#b22222")
        else:
            colors.append(bar_color_for_proto(proto_class_index(proto), gt_idx))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(title, fontsize=11)
    if delta_mode:
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("Delta contribution")
    else:
        ax.set_xlabel("Contribution")


def render_patch_grid(
    fig,
    outer_spec,
    proto_results: Sequence[Dict],
    prototype_img_dir: Path,
    score_key: str,
) -> None:
    patch_spec = outer_spec.subgridspec(2, 5, wspace=0.05, hspace=0.15)
    for index in range(10):
        ax = fig.add_subplot(patch_spec[index // 5, index % 5])
        if index < len(proto_results):
            proto = proto_results[index]
            patch = load_prototype_patch(prototype_img_dir, proto["proto_idx"])
            if patch is not None:
                ax.imshow(patch)
            else:
                ax.text(0.5, 0.5, f"P{proto['proto_idx']}", ha="center", va="center")
            ax.set_title(
                f"P{proto['proto_idx']}\n{score_key}={proto.get(score_key, 0.0):.2f}",
                fontsize=8,
            )
        ax.axis("off")


def render_narrative_figure(
    sample: Dict,
    image_tensor: torch.Tensor,
    unadapted_meta: Dict,
    prototta_meta: Dict,
    output_path: Path,
    prototype_img_dir: Path,
) -> None:
    ensure_dir(output_path.parent)
    img_rgb = tensor_to_rgb(image_tensor)

    fig = plt.figure(figsize=(22, 11))
    outer = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.75, 1.25], width_ratios=[1.0, 0.18, 1.0])

    left_spec = outer[:2, 0].subgridspec(3, 1, height_ratios=[1.25, 0.85, 1.35], hspace=0.3)
    right_spec = outer[:2, 2].subgridspec(3, 1, height_ratios=[1.25, 0.85, 1.35], hspace=0.3)

    left_img_ax = fig.add_subplot(left_spec[0])
    left_img_ax.imshow(img_rgb)
    unadapted_pred = unadapted_meta["predicted_top_prototypes"]
    if unadapted_pred:
        left_img_ax.imshow(create_heatmap(unadapted_pred[0], image_size=img_rgb.shape[0]), alpha=0.55, cmap="hot")
    left_img_ax.set_title(
        f"Unadapted: {unadapted_meta['predicted_class']} ({'correct' if unadapted_meta['is_correct'] else 'wrong'})",
        fontsize=12,
    )
    left_img_ax.axis("off")

    left_bar_ax = fig.add_subplot(left_spec[1])
    plot_prototype_bars(
        left_bar_ax,
        unadapted_meta["predicted_top_prototypes"],
        int(sample["ground_truth_index"]),
        delta_mode=False,
        title="Top-5 predicted-class contributions",
    )

    render_patch_grid(
        fig,
        left_spec[2],
        unadapted_meta["any_class_top_prototypes"],
        prototype_img_dir,
        "activation",
    )

    right_img_ax = fig.add_subplot(right_spec[0])
    right_img_ax.imshow(img_rgb)
    prototta_pred = prototta_meta["predicted_top_prototypes"]
    if prototta_pred:
        right_img_ax.imshow(create_heatmap(prototta_pred[0], image_size=img_rgb.shape[0]), alpha=0.55, cmap="hot")
    right_img_ax.set_title(
        f"ProtoTTA: {prototta_meta['predicted_class']} ({'correct' if prototta_meta['is_correct'] else 'wrong'})",
        fontsize=12,
    )
    right_img_ax.axis("off")

    delta_map = {
        proto["proto_idx"]: float(proto.get("contribution", 0.0))
        for proto in unadapted_meta["predicted_top_prototypes"]
    }
    delta_protos = []
    for proto in prototta_meta["predicted_top_prototypes"]:
        cloned = clone_proto(proto)
        cloned["delta_contribution"] = float(proto.get("contribution", 0.0)) - delta_map.get(
            proto["proto_idx"], 0.0
        )
        delta_protos.append(cloned)

    right_bar_ax = fig.add_subplot(right_spec[1])
    plot_prototype_bars(
        right_bar_ax,
        delta_protos,
        int(sample["ground_truth_index"]),
        delta_mode=True,
        title="Delta contribution vs unadapted",
    )

    render_patch_grid(
        fig,
        right_spec[2],
        prototta_meta["any_class_top_prototypes"],
        prototype_img_dir,
        "activation",
    )

    center_ax = fig.add_subplot(outer[:2, 1])
    center_ax.axis("off")
    center_ax.annotate(
        "",
        xy=(0.88, 0.5),
        xytext=(0.12, 0.5),
        arrowprops=dict(arrowstyle="simple", fc="#4f6d7a", ec="#4f6d7a", alpha=0.9),
    )
    center_ax.text(
        0.5,
        0.62,
        "ProtoTTA\nadaptation",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    caption_ax = fig.add_subplot(outer[2, :])
    caption_ax.set_facecolor("#f4f4f4")
    caption_ax.text(
        0.02,
        0.8,
        "Caption area reserved for VLM narrative text",
        fontsize=12,
        style="italic",
    )
    caption_ax.set_xticks([])
    caption_ax.set_yticks([])
    caption_ax.set_frame_on(True)

    fig.suptitle(
        f"{sample['corruption_type']} | GT: {sample['ground_truth_class']} | sample_idx={sample['sample_idx']}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def proto_to_json(proto: Dict, class_names: Sequence[str]) -> Dict:
    return {
        "proto_idx": int(proto["proto_idx"]),
        "activation": float(proto["activation"]),
        "class_index": int(proto["class"]),
        "class_name": humanize_class_name(class_names[int(proto["class"])]),
        "connection_weight": float(proto["connection_weight"]),
        "contribution": float(proto.get("contribution", 0.0)),
        "patch_locations": [
            [int(item) for item in np.asarray(proto["patch_locations"][0]).tolist()],
            [int(item) for item in np.asarray(proto["patch_locations"][1]).tolist()],
        ],
        "slots": [float(item) for item in np.asarray(proto["slots"]).tolist()],
    }


def sample_artifact_paths(results_dir: Path, method: str, sample: Dict) -> Dict[str, Path]:
    method_dir = results_dir / method
    stem = sample_stem(sample)
    sample_dir = method_dir / "samples" / stem
    return {
        "method_dir": method_dir,
        "sample_dir": sample_dir,
        "raw_image": sample_dir / "00_corrupted_input.png",
        "figure": sample_dir / "01_predicted_class_reasoning.png",
        "any_class_figure": sample_dir / "02_any_class_reasoning.png",
        "meta": sample_dir / "03_meta.json",
        "vlm_prompt": sample_dir / "04_vlm_prompt.txt",
        "vlm": sample_dir / "05_vlm.json",
        "vlm_raw": sample_dir / "05_vlm_raw.txt",
        "vlm_error": sample_dir / "05_vlm_error.txt",
        "vlm_parse_error": sample_dir / "05_vlm_parse_error.txt",
        "vlm_extracted_json": sample_dir / "05_vlm_extracted_json.txt",
    }


def find_json_candidate(raw_text: str) -> str:
    stripped = raw_text.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", stripped):
        start = match.start()
        try:
            parsed, end = decoder.raw_decode(stripped[start:])
            if isinstance(parsed, dict):
                candidates.append(stripped[start : start + end])
        except json.JSONDecodeError:
            continue
    return candidates[-1] if candidates else stripped


def extract_json_fragment(raw_text: str) -> Dict:
    candidate = find_json_candidate(raw_text)
    if not candidate:
        raise ValueError("VLM returned empty text, so there is no JSON to parse.")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        error = json.JSONDecodeError(exc.msg, exc.doc, exc.pos)
        setattr(error, "json_candidate", candidate)
        raise error


class VLMScorer:
    """Lazy Qwen3-VL scorer with basic JSON retry handling."""

    def __init__(self, model_id: str, max_new_tokens: int):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.process_vision_info = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        LOGGER.info("Loading VLM: %s", self.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.process_vision_info = process_vision_info

    def _prepare_inputs(self, image_paths: Sequence[Path], prompt: str):
        self._ensure_loaded()
        LOGGER.info("Preparing VLM inputs for %d image(s)", len(image_paths))
        content = [{"type": "image", "image": str(path)} for path in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if self.process_vision_info is not None:
                image_inputs, video_inputs = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                pil_images = [Image.open(path).convert("RGB") for path in image_paths]
                inputs = self.processor(
                    text=[text],
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                )
        target_device = (
            self.model.device
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )
        return inputs.to(target_device)

    def generate_json(self, image_paths: Sequence[Path], prompt: str) -> Tuple[Dict, str, str]:
        retry_prompt = (
            prompt
            + "\n\nReturn exactly one JSON object starting with '{' and ending with '}'. "
            "Do not include prose, markdown fences, or thinking traces."
        )
        last_error = None
        last_raw_text = ""
        last_candidate = ""
        last_prompt = prompt
        for attempt, current_prompt in enumerate([prompt, retry_prompt], start=1):
            try:
                last_prompt = current_prompt
                LOGGER.info(
                    "VLM generation attempt %d/2 starting for %d image(s)",
                    attempt,
                    len(image_paths),
                )
                inputs = self._prepare_inputs(image_paths, current_prompt)
                LOGGER.info("Prompt sent to VLM, waiting for generation...")
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                LOGGER.info("VLM generation finished, decoding output")
                trimmed_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw_text = self.processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                last_raw_text = raw_text
                last_candidate = find_json_candidate(raw_text)
                LOGGER.info(
                    "VLM returned %d characters; attempting JSON extraction",
                    len(raw_text),
                )
                return extract_json_fragment(raw_text), raw_text, current_prompt
            except Exception as exc:
                last_error = exc
                if not last_candidate and last_raw_text:
                    last_candidate = find_json_candidate(last_raw_text)
                LOGGER.warning("VLM generation attempt %d failed: %s", attempt, exc)
                if attempt == 2:
                    error = RuntimeError(str(exc))
                    setattr(error, "raw_text", last_raw_text)
                    setattr(error, "json_candidate", last_candidate)
                    setattr(error, "prompt_text", current_prompt)
                    raise error
        error = RuntimeError(f"Unreachable VLM retry state: {last_error}")
        setattr(error, "raw_text", last_raw_text)
        setattr(error, "json_candidate", last_candidate)
        setattr(error, "prompt_text", last_prompt)
        raise error


def call_vlm_for_part_a(
    scorer: VLMScorer,
    sample: Dict,
    sample_meta: Dict,
    artifacts: Dict[str, Path],
    corrupted_dir: Path,
    severity: int,
) -> str:
    if artifacts["vlm"].exists():
        return artifacts["vlm_prompt"].read_text(encoding="utf-8") if artifacts["vlm_prompt"].exists() else ""

    image_paths = [
        artifacts["raw_image"] if artifacts["raw_image"].exists() else sample_corrupted_image_path(sample, corrupted_dir, severity),
        artifacts["figure"],
        artifacts["any_class_figure"],
    ]
    prompt = (
        "You are evaluating the quality of a prototype-based bird classifier's reasoning.\n"
        "The model classifies birds using 'this looks like that' logic - it matches regions "
        "of the input image to learned prototype patches from training.\n\n"
        "You will receive THREE IMAGES in this exact order:\n"
        "IMAGE 1 = the raw corrupted test image.\n"
        "IMAGE 2 = the predicted-class reasoning board.\n"
        "  - Panel A = raw test image.\n"
        "  - Panel B = same image with a heatmap of the strongest matched prototype.\n"
        "  - Panels C1-C5 = TRAINING prototype patches retrieved by the model.\n"
        "  - The C-panels are NOT crops from the test image; they are stored prototype exemplars from training.\n"
        "  - 'Pxxx' is the prototype id and the last number under each patch is its contribution score.\n"
        "  - Colored square outlines inside each training prototype patch mark the active prototype slots/sub-patches.\n"
        "IMAGE 3 = the any-class reasoning board.\n"
        "  - Same A/B/C layout as IMAGE 2, but Panels C1-C10 are the strongest prototypes contributing to ANY class, "
        "not only the predicted class.\n"
        "  - Use IMAGE 3 to detect spurious wrong-class evidence.\n\n"
        f"The model predicted: {sample_meta['predicted_class']}\n"
        f"The correct answer is: {sample_meta['ground_truth_class']}\n"
        f"This prediction is: {'CORRECT' if sample_meta['is_correct'] else 'WRONG'}\n\n"
        "Judge the model mostly from IMAGE 2. Use IMAGE 1 for the actual corrupted appearance of the bird, "
        "and IMAGE 3 for broader any-class evidence.\n\n"
        "Please rate the SEMANTIC QUALITY of this model's reasoning on two dimensions:\n\n"
        "PART_COHERENCE_SCORE (1-5):\n"
        "  1 = model focused entirely on background, noise, or artifacts\n"
        "  2 = model focused mostly on non-bird or uninformative regions\n"
        "  3 = model focused on bird but non-discriminative part (belly, back)\n"
        "  4 = model focused on a meaningful bird part (wing, tail, breast pattern)\n"
        "  5 = model focused on a highly discriminative part (beak, eye, crown, species-specific marking)\n\n"
        "PROTOTYPE_MATCH_SCORE (1-5):\n"
        "  1 = prototype patches look nothing like the highlighted image region\n"
        "  2 = very weak visual similarity\n"
        "  3 = moderate similarity (same general region/color but different details)\n"
        "  4 = good similarity (clear visual match)\n"
        "  5 = excellent similarity (the prototype patch is clearly the same type of feature as what was highlighted)\n\n"
        "OVERALL_ADAPTATION_QUALITY (1-5):\n"
        "  1 = reasoning is completely incoherent\n"
        "  3 = reasoning is plausible but not convincing\n"
        "  5 = reasoning is semantically coherent and would convince an ornithologist\n\n"
        "Respond in JSON only with keys: part_coherence_score, prototype_match_score, "
        "overall_adaptation_quality, part_name, one_sentence_summary.\n"
        "Return exactly one JSON object and nothing else."
    )

    artifacts["vlm_prompt"].write_text(prompt, encoding="utf-8")
    LOGGER.info("Calling VLM for Part A sample %s", sample_stem(sample))
    parsed, raw_text, _ = scorer.generate_json(image_paths, prompt)
    artifacts["vlm_raw"].write_text(raw_text, encoding="utf-8")
    payload = dict(parsed)
    payload["raw_response"] = raw_text
    write_json(artifacts["vlm"], payload)
    LOGGER.info("Saved VLM JSON for Part A sample %s", sample_stem(sample))
    return prompt


def call_vlm_for_narrative(
    scorer: VLMScorer,
    sample: Dict,
    unadapted_meta: Dict,
    prototta_meta: Dict,
    figure_path: Path,
    narrative_path: Path,
) -> str:
    if narrative_path.exists():
        prompt_path = narrative_path.with_name("01_narrative_prompt.txt")
        return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    prompt = (
        "You are an expert ornithologist analyzing how a deep learning model's reasoning "
        "changed after test-time adaptation.\n\n"
        "You will receive ONE composite figure with this structure:\n"
        "- LEFT column: unadapted model evidence\n"
        "- CENTER: an arrow labeled ProtoTTA adaptation\n"
        "- RIGHT column: ProtoTTA-adapted model evidence\n"
        "- Within each side: top row = raw corrupted image plus focus heatmap, "
        "middle row = prototype contribution bars, bottom row = training prototype patches matched by the model\n\n"
        f"I will show you side-by-side prototype evidence for two model states:\n"
        f"LEFT: The UNADAPTED model (predicted {unadapted_meta['predicted_class']} - WRONG)\n"
        f"RIGHT: The ProtoTTA ADAPTED model (predicted {prototta_meta['predicted_class']} - CORRECT)\n"
        f"Ground truth: {sample['ground_truth_class']}\n\n"
        f"The corruption type applied to this image was: {sample['corruption_type']}\n\n"
        "For each model, you can see:\n"
        "- Which image region it focused on (heatmap)\n"
        "- Which prototype patches from training it matched\n"
        "- The contribution strength of each prototype\n\n"
        "Please provide:\n\n"
        "BEFORE_DESCRIPTION: In 1-2 sentences, describe what visual evidence the UNADAPTED "
        "model was using and why it led to a wrong prediction. Be specific about bird anatomy.\n\n"
        "AFTER_DESCRIPTION: In 1-2 sentences, describe what visual evidence the ADAPTED model "
        "used and why it led to the correct prediction. Be specific about bird anatomy.\n\n"
        "CHANGE_SUMMARY: Complete this sentence in 15 words or less:\n"
        f"'ProtoTTA shifted the model's attention from ___ to ___, enabling correct identification "
        f"of the {sample['ground_truth_class']}.'\n\n"
        "ADAPTATION_MECHANISM: Which best describes what happened?\n"
        "  A) Attention shifted from noise/artifacts to real bird features\n"
        "  B) Attention shifted from one bird part to a more discriminative bird part\n"
        "  C) Prototype matching quality improved for the same region\n"
        "  D) Spurious wrong-class prototypes were suppressed\n\n"
        "Respond in JSON with keys: before_description, after_description, change_summary, "
        "adaptation_mechanism, confidence."
    )

    narrative_path.with_name("01_narrative_prompt.txt").write_text(prompt, encoding="utf-8")
    LOGGER.info("Calling VLM for narrative sample %s", sample_stem(sample))
    parsed, raw_text, _ = scorer.generate_json([figure_path], prompt)
    narrative_path.with_name("02_narrative_raw.txt").write_text(raw_text, encoding="utf-8")
    payload = dict(parsed)
    payload["raw_response"] = raw_text
    write_json(narrative_path, payload)
    LOGGER.info("Saved VLM narrative JSON for sample %s", sample_stem(sample))
    return prompt


def method_complete(
    results_dir: Path, method: str, samples: Sequence[Dict], require_vlm: bool
) -> bool:
    for sample in samples:
        artifacts = sample_artifact_paths(results_dir, method, sample)
        required = [
            artifacts["raw_image"],
            artifacts["meta"],
            artifacts["figure"],
            artifacts["any_class_figure"],
        ]
        if require_vlm:
            required.extend([artifacts["vlm_prompt"], artifacts["vlm"]])
        if any(not path.exists() for path in required):
            return False
    return True


def build_sample_meta_payload(
    method: str,
    sample: Dict,
    artifacts: Dict[str, Path],
    pred_idx: int,
    predicted_top: Sequence[Dict],
    any_class_top: Sequence[Dict],
    class_names: Sequence[str],
    use_precomputed: bool,
) -> Dict:
    return {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "sample_dir": str(artifacts["sample_dir"]),
        "sample_idx": int(sample["sample_idx"]),
        "subset_position": int(sample["subset_position"]),
        "corruption_type": sample["corruption_type"],
        "image_path": sample["image_path"],
        "ground_truth_class": sample["ground_truth_class"],
        "ground_truth_index": int(sample["ground_truth_index"]),
        "predicted_class": humanize_class_name(class_names[pred_idx]),
        "predicted_index": pred_idx,
        "is_correct": bool(pred_idx == sample["ground_truth_index"]),
        "raw_image_path": str(artifacts["raw_image"]),
        "figure_path": str(artifacts["figure"]),
        "any_class_figure_path": str(artifacts["any_class_figure"]),
        "vlm_prompt_path": str(artifacts["vlm_prompt"]),
        "vlm_response_path": str(artifacts["vlm"]),
        "predicted_top_prototypes": [
            proto_item if use_precomputed else proto_to_json(proto_item, class_names)
            for proto_item in predicted_top
        ],
        "any_class_top_prototypes": [
            proto_item if use_precomputed else proto_to_json(proto_item, class_names)
            for proto_item in any_class_top
        ],
    }


def materialize_sample_artifacts(
    method: str,
    sample: Dict,
    artifacts: Dict[str, Path],
    image_tensor: torch.Tensor,
    pred_idx: int,
    predicted_top: Sequence[Dict],
    any_class_top: Sequence[Dict],
    prototype_img_dir: Path,
    class_names: Sequence[str],
    use_precomputed: bool,
) -> Dict:
    ensure_dir(artifacts["sample_dir"])
    if not artifacts["raw_image"].exists():
        save_rgb_image(image_tensor, artifacts["raw_image"])

    title = (
        f"{DISPLAY_NAMES[method]} | Pred: {humanize_class_name(class_names[pred_idx])} | "
        f"GT: {sample['ground_truth_class']} | {'CORRECT' if pred_idx == sample['ground_truth_index'] else 'WRONG'}"
    )
    if not artifacts["figure"].exists():
        render_reasoning_overview(
            image_tensor=image_tensor,
            proto_results=predicted_top,
            title=title,
            output_path=artifacts["figure"],
            prototype_img_dir=prototype_img_dir,
            grid_rows=2,
            grid_cols=3,
            class_names=class_names,
            board_label="Predicted-class reasoning board",
        )
    if not artifacts["any_class_figure"].exists():
        render_reasoning_overview(
            image_tensor=image_tensor,
            proto_results=any_class_top,
            title=f"{title} | Any-class contributions",
            output_path=artifacts["any_class_figure"],
            prototype_img_dir=prototype_img_dir,
            grid_rows=2,
            grid_cols=5,
            class_names=class_names,
            board_label="Any-class reasoning board",
        )

    sample_meta = build_sample_meta_payload(
        method=method,
        sample=sample,
        artifacts=artifacts,
        pred_idx=pred_idx,
        predicted_top=predicted_top,
        any_class_top=any_class_top,
        class_names=class_names,
        use_precomputed=use_precomputed,
    )
    write_json(artifacts["meta"], sample_meta)
    return sample_meta


def run_method_evaluation(
    method: str,
    manifest_samples: Sequence[Dict],
    args: argparse.Namespace,
    device: torch.device,
    prototype_img_dir: Path,
    class_names: Sequence[str],
    scorer: Optional[VLMScorer],
    require_vlm: bool,
    eata_fishers: Optional[Dict],
) -> None:
    if method_complete(args.results_dir, method, manifest_samples, require_vlm=require_vlm):
        LOGGER.info("Skipping %s because all artifacts already exist", DISPLAY_NAMES[method])
        return

    use_precomputed = method in BATCH_PRECOMPUTE_METHODS
    transform = build_transform()
    method_dir = args.results_dir / method
    ensure_dir(method_dir)
    samples_to_process = manifest_samples[: args.max_samples] if args.max_samples else manifest_samples
    if use_precomputed:
        pbar = tqdm(samples_to_process, desc=DISPLAY_NAMES[method], leave=False)
        for sample in pbar:
            artifacts = sample_artifact_paths(args.results_dir, method, sample)
            image_path = sample_corrupted_image_path(sample, args.corrupted_dir, args.severity)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = transform(image)
            precomputed_entry = load_precomputed_entry(args.results_dir, method, sample)
            if precomputed_entry is None:
                raise RuntimeError(
                    f"Missing precomputed entry for {DISPLAY_NAMES[method]} / {sample_stem(sample)}"
                )
            pred_idx = int(precomputed_entry["predicted_index"])
            predicted_top = precomputed_entry["predicted_top_prototypes"]
            any_class_top = precomputed_entry["any_class_top_prototypes"]
            sample_meta = materialize_sample_artifacts(
                method=method,
                sample=sample,
                artifacts=artifacts,
                image_tensor=image_tensor,
                pred_idx=pred_idx,
                predicted_top=predicted_top,
                any_class_top=any_class_top,
                prototype_img_dir=prototype_img_dir,
                class_names=class_names,
                use_precomputed=True,
            )
            if require_vlm and scorer is not None:
                try:
                    LOGGER.info("Starting VLM scoring for %s / %s", DISPLAY_NAMES[method], sample_stem(sample))
                    prompt_text = call_vlm_for_part_a(
                        scorer=scorer,
                        sample=sample,
                        sample_meta=sample_meta,
                        artifacts=artifacts,
                        corrupted_dir=args.corrupted_dir,
                        severity=args.severity,
                    )
                    if prompt_text and not artifacts["vlm_prompt"].exists():
                        artifacts["vlm_prompt"].write_text(prompt_text, encoding="utf-8")
                except Exception as exc:
                    LOGGER.exception("VLM call failed for %s / %s", DISPLAY_NAMES[method], sample_stem(sample))
                    raw_text = getattr(exc, "raw_text", "")
                    if raw_text:
                        artifacts["vlm_raw"].write_text(raw_text, encoding="utf-8")
                    json_candidate = getattr(exc, "json_candidate", "")
                    if json_candidate:
                        artifacts["vlm_extracted_json"].write_text(json_candidate, encoding="utf-8")
                    prompt_text = getattr(exc, "prompt_text", "")
                    if prompt_text and not artifacts["vlm_prompt"].exists():
                        artifacts["vlm_prompt"].write_text(prompt_text, encoding="utf-8")
                    artifacts["vlm_error"].write_text(str(exc), encoding="utf-8")
                    artifacts["vlm_parse_error"].write_text(repr(exc), encoding="utf-8")
        return

    targets_by_corruption: Dict[str, Dict[str, Dict]] = {}
    for sample in samples_to_process:
        artifacts = sample_artifact_paths(args.results_dir, method, sample)
        required = [artifacts["raw_image"], artifacts["meta"], artifacts["figure"], artifacts["any_class_figure"]]
        if require_vlm:
            required.extend([artifacts["vlm_prompt"], artifacts["vlm"]])
        if all(path.exists() for path in required):
            continue
        targets_by_corruption.setdefault(sample["corruption_type"], {})[
            precompute_key(sample["corruption_type"], sample["image_path"])
        ] = {
            "sample": sample,
            "artifacts": artifacts,
        }

    for corruption_type in CORRUPTION_TYPES:
        target_map = targets_by_corruption.get(corruption_type, {})
        if not target_map:
            continue

        eval_model = setup_method_model(method, args.model, device, args, samples_to_process, eata_fishers)
        ppnet = get_underlying_model(eval_model)
        ppnet.eval()

        dataset = load_corruption_dataset(args.corrupted_dir, corruption_type, args.severity)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        cursor = 0
        pbar = tqdm(loader, desc=f"{DISPLAY_NAMES[method]}:{corruption_type}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = eval_model(images)
            logits, min_distances, values = normalize_outputs(outputs)
            preds = logits.argmax(dim=1)

            batch_targets = []
            for batch_index in range(labels.size(0)):
                dataset_index = cursor + batch_index
                sample_path, _ = dataset.samples[dataset_index]
                rel_path = os.path.relpath(sample_path, dataset.root)
                key = precompute_key(corruption_type, rel_path)
                if key not in target_map:
                    continue
                batch_targets.append((batch_index, target_map[key]["sample"], target_map[key]["artifacts"]))
            cursor += labels.size(0)

            if not batch_targets:
                continue

            batch_proto_results = viz.get_top_k_prototypes_batch(
                ppnet,
                images,
                k=None,
                precomputed_outputs=(logits, min_distances, values),
                sort_by="activation",
            )

            for batch_index, sample, artifacts in batch_targets:
                all_proto_results, pred_class_from_proto = batch_proto_results[batch_index]
                pred_idx = int(preds[batch_index].item())
                if pred_idx != pred_class_from_proto:
                    LOGGER.warning(
                        "%s prediction mismatch on %s: logits=%d proto=%d",
                        DISPLAY_NAMES[method],
                        sample_stem(sample),
                        pred_idx,
                        pred_class_from_proto,
                    )

                predicted_top = select_predicted_class_prototypes(all_proto_results, pred_idx, ppnet, top_k=5)
                any_class_top = select_any_class_prototypes(all_proto_results, ppnet, top_k=10)
                image_tensor = images[batch_index].detach().cpu()
                sample_meta = materialize_sample_artifacts(
                    method=method,
                    sample=sample,
                    artifacts=artifacts,
                    image_tensor=image_tensor,
                    pred_idx=pred_idx,
                    predicted_top=predicted_top,
                    any_class_top=any_class_top,
                    prototype_img_dir=prototype_img_dir,
                    class_names=class_names,
                    use_precomputed=False,
                )

                if require_vlm and scorer is not None:
                    try:
                        LOGGER.info("Starting VLM scoring for %s / %s", DISPLAY_NAMES[method], sample_stem(sample))
                        prompt_text = call_vlm_for_part_a(
                            scorer=scorer,
                            sample=sample,
                            sample_meta=sample_meta,
                            artifacts=artifacts,
                            corrupted_dir=args.corrupted_dir,
                            severity=args.severity,
                        )
                        if prompt_text and not artifacts["vlm_prompt"].exists():
                            artifacts["vlm_prompt"].write_text(prompt_text, encoding="utf-8")
                    except Exception as exc:
                        LOGGER.exception("VLM call failed for %s / %s", DISPLAY_NAMES[method], sample_stem(sample))
                        raw_text = getattr(exc, "raw_text", "")
                        if raw_text:
                            artifacts["vlm_raw"].write_text(raw_text, encoding="utf-8")
                        json_candidate = getattr(exc, "json_candidate", "")
                        if json_candidate:
                            artifacts["vlm_extracted_json"].write_text(json_candidate, encoding="utf-8")
                        prompt_text = getattr(exc, "prompt_text", "")
                        if prompt_text and not artifacts["vlm_prompt"].exists():
                            artifacts["vlm_prompt"].write_text(prompt_text, encoding="utf-8")
                        artifacts["vlm_error"].write_text(str(exc), encoding="utf-8")
                        artifacts["vlm_parse_error"].write_text(repr(exc), encoding="utf-8")

        del eval_model
        torch.cuda.empty_cache()


def aggregate_scores(results_dir: Path, manifest_samples: Sequence[Dict]) -> Dict[str, Dict]:
    aggregate: Dict[str, Dict] = {}
    for method in METHOD_ORDER:
        overall_scores = []
        part_scores = []
        proto_scores = []
        correct_scores = []
        failed_samples = []
        for sample in manifest_samples:
            artifacts = sample_artifact_paths(results_dir, method, sample)
            if not artifacts["meta"].exists():
                continue
            if not artifacts["vlm"].exists():
                if artifacts["vlm_error"].exists():
                    failed_samples.append(
                        {
                            "sample": sample_stem(sample),
                            "error": artifacts["vlm_error"].read_text(encoding="utf-8"),
                        }
                    )
                continue
            vlm_payload = load_json(artifacts["vlm"])
            meta_payload = load_json(artifacts["meta"])
            overall = float(vlm_payload["overall_adaptation_quality"])
            part = float(vlm_payload["part_coherence_score"])
            proto = float(vlm_payload["prototype_match_score"])
            overall_scores.append(overall)
            part_scores.append(part)
            proto_scores.append(proto)
            if meta_payload["is_correct"]:
                correct_scores.append(overall)

        if overall_scores:
            aggregate[method] = {
                "display_name": DISPLAY_NAMES[method],
                "num_scored": len(overall_scores),
                "VAQ": float(np.mean(overall_scores)),
                "VAQ_correct": float(np.mean(correct_scores)) if correct_scores else None,
                "VAQ_std": float(np.std(overall_scores, ddof=0)),
                "part_coherence_mean": float(np.mean(part_scores)),
                "prototype_match_mean": float(np.mean(proto_scores)),
                "num_failed": len(failed_samples),
                "failed_samples": failed_samples,
            }
        else:
            aggregate[method] = {
                "display_name": DISPLAY_NAMES[method],
                "num_scored": 0,
                "VAQ": None,
                "VAQ_correct": None,
                "VAQ_std": None,
                "part_coherence_mean": None,
                "prototype_match_mean": None,
                "num_failed": len(failed_samples),
                "failed_samples": failed_samples,
            }
    return aggregate


def write_aggregate_payload(results_dir: Path, manifest_samples: Sequence[Dict], severity: int) -> Dict[str, Dict]:
    scores = aggregate_scores(results_dir, manifest_samples)
    summary_table = render_summary_table(scores)
    payload = {
        "severity": severity,
        "methods": scores,
        "summary_table": summary_table,
    }
    write_json(results_dir / "vaq_scores.json", payload)

    for method, method_payload in scores.items():
        method_dir = results_dir / method
        if method_dir.exists():
            write_json(
                method_dir / "method_scores.json",
                {
                    "severity": severity,
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "scores": method_payload,
                },
            )
            write_json(
                method_dir / "failed_vlm_samples.json",
                {
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "num_failed": method_payload.get("num_failed", 0),
                    "failed_samples": method_payload.get("failed_samples", []),
                },
            )
    return scores


def render_summary_table(scores: Dict[str, Dict]) -> str:
    lines = [
        "Method        | VAQ ↑  | Part Coh. ↑ | Proto Match ↑",
        "------------- | ------ | ----------- | -------------",
    ]
    for method in TABLE_METHOD_ORDER:
        payload = scores[method]
        if payload["VAQ"] is None:
            lines.append(f"{payload['display_name']:<13} | n/a    | n/a         | n/a")
            continue
        lines.append(
            f"{payload['display_name']:<13} | "
            f"{payload['VAQ']:.2f}   | "
            f"{payload['part_coherence_mean']:.2f}        | "
            f"{payload['prototype_match_mean']:.2f}"
        )
    return "\n".join(lines)


def run_part_a(
    manifest: Dict,
    args: argparse.Namespace,
    device: torch.device,
    prototype_img_dir: Path,
    class_names: Sequence[str],
) -> None:
    manifest_samples = manifest["samples"][: args.max_samples] if args.max_samples else manifest["samples"]
    selected_methods = [args.method] if args.method else METHOD_ORDER
    scorer = VLMScorer(args.vlm_model_id, args.max_new_tokens)
    eata_fishers = None

    for method in selected_methods:
        if method in BATCH_PRECOMPUTE_METHODS:
            run_full_dataset_precompute(
                method=method,
                args=args,
                device=device,
                prototype_img_dir=prototype_img_dir,
                class_names=class_names,
                eata_fishers=eata_fishers,
            )
        if method == "eata" and not method_complete(
            args.results_dir, method, manifest_samples, require_vlm=True
        ):
            eata_fishers = compute_eata_fishers(args.model, device, args, manifest_samples)
        run_method_evaluation(
            method=method,
            manifest_samples=manifest_samples,
            args=args,
            device=device,
            prototype_img_dir=prototype_img_dir,
            class_names=class_names,
            scorer=scorer,
            require_vlm=True,
            eata_fishers=eata_fishers,
        )

    scores = write_aggregate_payload(args.results_dir, manifest_samples, args.severity)
    if args.method is None:
        print(render_summary_table(scores))


def select_narrative_candidates(results_dir: Path, manifest_samples: Sequence[Dict]) -> List[Dict]:
    candidates = []
    for sample in manifest_samples:
        unadapted_artifacts = sample_artifact_paths(results_dir, "unadapted", sample)
        prototta_artifacts = sample_artifact_paths(results_dir, "prototta", sample)
        if not (
            unadapted_artifacts["meta"].exists()
            and prototta_artifacts["meta"].exists()
            and prototta_artifacts["vlm"].exists()
        ):
            continue

        unadapted_meta = load_json(unadapted_artifacts["meta"])
        prototta_meta = load_json(prototta_artifacts["meta"])
        prototta_vlm = load_json(prototta_artifacts["vlm"])

        if (
            prototta_meta["is_correct"]
            and not unadapted_meta["is_correct"]
            and float(prototta_vlm["overall_adaptation_quality"]) >= 4.0
        ):
            candidates.append(
                {
                    "sample": sample,
                    "unadapted_meta": unadapted_meta,
                    "prototta_meta": prototta_meta,
                    "prototta_vlm": prototta_vlm,
                }
            )

    candidates.sort(
        key=lambda item: (
            float(item["prototta_vlm"]["overall_adaptation_quality"]),
            float(item["prototta_vlm"]["prototype_match_score"]),
        ),
        reverse=True,
    )

    selected = []
    seen_corruptions = set()
    for candidate in candidates:
        corruption_type = candidate["sample"]["corruption_type"]
        if corruption_type in seen_corruptions:
            continue
        selected.append(candidate)
        seen_corruptions.add(corruption_type)
        if len(selected) == 10:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == 10:
            break
    return selected


def write_narrative_summary(
    narratives_dir: Path,
    selected: Sequence[Dict],
) -> None:
    counts = Counter()
    lines = ["# ProtoTTA Narrative Summary", ""]
    for index, item in enumerate(selected, start=1):
        sample = item["sample"]
        stem = sample_stem(sample)
        figure_name = f"{stem}_figure.png"
        narrative_path = narratives_dir / f"{stem}_narrative.json"
        narrative = load_json(narrative_path) if narrative_path.exists() else {}
        mechanism = narrative.get("adaptation_mechanism")
        if mechanism:
            counts[mechanism] += 1

        lines.extend(
            [
                f"## Sample {index}: {sample['corruption_type']} / {sample['ground_truth_class']}",
                "",
                f"![{stem}]({figure_name})",
                "",
                f"*{narrative.get('change_summary', 'Narrative pending.')}*",
                "",
            ]
        )

    lines.extend(
        [
            "## Adaptation Mechanism Distribution",
            "",
            f"- A: {counts.get('A', 0)}",
            f"- B: {counts.get('B', 0)}",
            f"- C: {counts.get('C', 0)}",
            f"- D: {counts.get('D', 0)}",
            "",
        ]
    )

    (narratives_dir / "narrative_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_part_b(
    manifest: Dict,
    args: argparse.Namespace,
    device: torch.device,
    prototype_img_dir: Path,
    class_names: Sequence[str],
) -> None:
    scorer = VLMScorer(args.vlm_model_id, args.max_new_tokens)
    eata_fishers = None

    for method, need_vlm in [("unadapted", False), ("prototta", True)]:
        if method in BATCH_PRECOMPUTE_METHODS:
            run_full_dataset_precompute(
                method=method,
                args=args,
                device=device,
                prototype_img_dir=prototype_img_dir,
                class_names=class_names,
                eata_fishers=eata_fishers,
            )
        run_method_evaluation(
            method=method,
            manifest_samples=manifest["samples"],
            args=args,
            device=device,
            prototype_img_dir=prototype_img_dir,
            class_names=class_names,
            scorer=scorer if need_vlm else None,
            require_vlm=need_vlm,
            eata_fishers=eata_fishers,
        )

    selected = select_narrative_candidates(args.results_dir, manifest["samples"])
    if not selected:
        raise RuntimeError(
            "No ProtoTTA narrative candidates met the requested criteria. "
            "Run Part A first or inspect the subset and thresholds."
        )

    narratives_dir = args.results_dir / "narratives"
    ensure_dir(narratives_dir)
    transform = build_transform()

    for item in tqdm(selected, desc="Narratives", leave=False):
        sample = item["sample"]
        corrupted_image_path = sample_corrupted_image_path(sample, args.corrupted_dir, args.severity)
        with Image.open(corrupted_image_path) as image:
            image = image.convert("RGB")
            image_tensor = transform(image)

        figure_path = narratives_dir / f"{sample_stem(sample)}_figure.png"
        if not figure_path.exists():
            render_narrative_figure(
                sample=sample,
                image_tensor=image_tensor,
                unadapted_meta=item["unadapted_meta"],
                prototta_meta=item["prototta_meta"],
                output_path=figure_path,
                prototype_img_dir=prototype_img_dir,
            )

        narrative_path = narratives_dir / f"{sample_stem(sample)}_narrative.json"
        try:
            call_vlm_for_narrative(
                scorer=scorer,
                sample=sample,
                unadapted_meta=item["unadapted_meta"],
                prototta_meta=item["prototta_meta"],
                figure_path=figure_path,
                narrative_path=narrative_path,
            )
        except Exception as exc:
            LOGGER.exception("Narrative VLM call failed for %s", sample_stem(sample))
            (narratives_dir / f"{sample_stem(sample)}_narrative_error.txt").write_text(
                str(exc), encoding="utf-8"
            )

    write_narrative_summary(narratives_dir, selected)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.model = resolve_path(args.model)
    args.clean_dir = resolve_path(args.clean_dir)
    args.corrupted_dir = resolve_path(args.corrupted_dir)
    args.results_dir = resolve_path(args.results_dir)
    args.prototype_dir = resolve_path(args.prototype_dir) if args.prototype_dir else None

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    if not torch.cuda.is_available():
        raise RuntimeError("This script expects CUDA because the ProtoViT utilities are CUDA-only.")
    device = torch.device("cuda")

    seed_everything(42)
    ensure_dir(args.results_dir)
    install_timm_checkpoint_compat()
    lazy_import_runtime_modules()

    prototype_img_dir = resolve_prototype_dir(args.model, args.prototype_dir)
    clean_dataset = load_clean_dataset(args.clean_dir)
    class_names = clean_dataset.classes
    subset_path = args.results_dir / "subset.json"

    if args.rebuild_subset and subset_path.exists():
        subset_path.unlink()

    if not precompute_complete(args.results_dir, "unadapted", args.severity):
        run_full_dataset_precompute(
            method="unadapted",
            args=args,
            device=device,
            prototype_img_dir=prototype_img_dir,
            class_names=class_names,
            eata_fishers=None,
        )

    manifest = build_subset_manifest(
        subset_path=subset_path,
        results_dir=args.results_dir,
        severity=args.severity,
        rebuild=args.rebuild_subset,
    )

    if args.part in (None, "A"):
        run_part_a(manifest, args, device, prototype_img_dir, class_names)
    if args.part in (None, "B"):
        if args.method is not None:
            LOGGER.warning("--method is ignored for Part B because narratives require both unadapted and ProtoTTA.")
        run_part_b(manifest, args, device, prototype_img_dir, class_names)


if __name__ == "__main__":
    main()
