#!/usr/bin/env python3
"""
run_inference_dogs.py
======================
Single-corruption/method inference for ProtoPFormer on Stanford Dogs-C.

Equivalent to ProtoViT's run_inference.py — run one method against one
corruption at one severity, and print accuracy.

Usage
-----
# Clean baseline
python run_inference_dogs.py \\
    --model output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth \\
    --mode normal

# Single corruption, normal inference
python run_inference_dogs.py \\
    --model  epoch-best.pth \\
    --mode   normal \\
    --corruption gaussian_noise \\
    --severity   3

# ProtoTTA (proto_imp_conf_v3)
python run_inference_dogs.py \\
    --model  epoch-best.pth \\
    --mode   proto_tta \\
    --corruption fog \\
    --severity   5

# Tent
python run_inference_dogs.py --model epoch-best.pth --mode tent \\
    --corruption contrast --severity 5

# EATA
python run_inference_dogs.py --model epoch-best.pth --mode eata \\
    --corruption brightness --severity 5
"""
import os
import sys
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import protopformer  # noqa: F401  needed for torch.load
from noise_utils import CORRUPTION_TYPES, get_corrupted_transform
from proto_tta import (
    setup_tent, setup_eata, setup_proto_tta,
    compute_fishers,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers (shared with evaluate_robustness_dogs.py)
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    """Load ProtoPFormer checkpoint.

    main.py saves checkpoints as:
        {'model': state_dict, 'args': argparse.Namespace, 'epoch': ..., ...}
    We reconstruct the architecture from the saved args, then load the weights.
    """
    import protopformer as ppf

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        # Rare case: full model saved directly
        return ckpt.to(device).eval()

    # Standard case: state-dict checkpoint
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError(f"Unexpected checkpoint format in {model_path}")

    # Reconstruct model from the args stored at save time
    args = ckpt['args']
    model = ppf.construct_PPNet(
        base_architecture=args.base_architecture,
        pretrained=False,          # weights come from checkpoint
        img_size=args.img_size,
        prototype_shape=args.prototype_shape,
        num_classes=args.nb_classes,
        reserve_layers=args.reserve_layers,
        reserve_token_nums=args.reserve_token_nums,
        use_global=args.use_global,
        use_ppc_loss=args.use_ppc_loss,
        ppc_cov_thresh=args.ppc_cov_thresh,
        ppc_mean_thresh=args.ppc_mean_thresh,
        global_coe=args.global_coe,
        global_proto_per_class=args.global_proto_per_class,
        prototype_activation_function=args.prototype_activation_function,
        add_on_layers_type=args.add_on_layers_type,
    )
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def get_loader(corruption_type, severity, data_dir, clean_dir,
               on_the_fly, batch_size, num_workers=4):
    # Auto-redirect dataset root → Images/ if needed (ImageFolder-compatible)
    p = Path(clean_dir)
    if (p / 'Images').is_dir():
        clean_dir = str(p / 'Images')
        logger.info(f"Redirected clean_dir → {clean_dir}")
    if corruption_type is None:
        # Clean
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        dataset = datasets.ImageFolder(clean_dir, transform)
    elif on_the_fly:
        transform = get_corrupted_transform(
            IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, corruption_type, severity)
        dataset = datasets.ImageFolder(clean_dir, transform)
    else:
        src = Path(data_dir) / corruption_type / str(severity)
        if not src.exists():
            raise FileNotFoundError(f"Pre-generated data not found: {src}\n"
                                    f"Run create_dogs_c.py first, or pass --on_the_fly")
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        dataset = datasets.ImageFolder(str(src), transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)


# ---------------------------------------------------------------------------
# Evaluate helpers
# ---------------------------------------------------------------------------

def eval_normal(model, loader, device):
    model.eval()
    n_correct = n_total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out    = model(images)
            logits = out[0] if isinstance(out, tuple) else out
            preds  = logits.argmax(dim=1)
            n_correct += preds.eq(labels).sum().item()
            n_total   += labels.size(0)
    return n_correct / n_total


def eval_tta(wrapper, loader, device):
    """TTA methods update the model on each batch during forward."""
    n_correct = n_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out    = wrapper(images)
        logits = out[0] if isinstance(out, tuple) else out
        preds  = logits.argmax(dim=1)
        n_correct += preds.eq(labels).sum().item()
        n_total   += labels.size(0)
    return n_correct / n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-method inference for ProtoPFormer on Dogs-C')

    parser.add_argument('--model', required=True, help='Path to epoch-best.pth')
    parser.add_argument('--modes', nargs='+', default=['normal', 'tent', 'eata', 'proto_tta'],
                        help='List of methods to evaluate')
    parser.add_argument('--corruption', default='gaussian_noise',
                        choices=CORRUPTION_TYPES + [None],
                        help='Corruption type (defaults to gaussian_noise)')
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--data_dir', default='datasets/stanford_dogs_c',
                        help='Pre-generated Dogs-C directory')
    parser.add_argument('--clean_dir', default=None,
                        help='Clean Dogs test dir (used for clean eval and on-the-fly)')
    parser.add_argument('--on_the_fly', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-3, help='Default LR for Tent/EATA')
    parser.add_argument('--proto_lr', type=float, default=1e-3, help='Specific LR for ProtoTTA')
    parser.add_argument('--proto_threshold', type=float, default=0.3, help='Geometric threshold for ProtoTTA reliability')
    parser.add_argument('--proto_mapping', type=str, default='sigmoid', choices=['sigmoid', 'linear'],
                        help='Similarity mapping used by ProtoTTA')
    parser.add_argument('--proto_sigmoid_center', type=float, default=2.0,
                        help='Center of sigmoid mapping in raw ProtoPFormer similarity units')
    parser.add_argument('--proto_sigmoid_temp', type=float, default=1.0,
                        help='Temperature of sigmoid mapping in raw ProtoPFormer similarity units')
    parser.add_argument('--proto_branch_agreement', action='store_true',
                        help='Require global/local branch agreement before adaptation')
    parser.add_argument('--proto_branch', type=str, default='both', choices=['local', 'global', 'both'],
                        help='Which prototype branch to use for ProtoTTA')
    parser.add_argument('--proto_plus', action='store_true',
                        help='Use ProtoTTA+ hybrid loss: prototype loss + logit entropy')
    parser.add_argument('--proto_weight', type=float, default=0.7,
                        help='Weight of prototype-aware loss in ProtoTTA+')
    parser.add_argument('--logit_weight', type=float, default=0.3,
                        help='Weight of logit entropy loss in ProtoTTA+')
    parser.add_argument('--proto_no_importance', action='store_true',
                        help='Disable prototype importance weighting in ProtoTTA')
    parser.add_argument('--proto_all_prototypes', action='store_true',
                        help='Sharpen all prototypes instead of only predicted-class prototypes')
    parser.add_argument('--steps', type=int, default=1)

    args = parser.parse_args()

    # --- Device ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clean_dir = args.clean_dir or str(Path(args.data_dir).parent / 'stanford_dogs' / 'Images')

    # --- Describe run ---
    desc = args.corruption if args.corruption else 'clean'
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Corruption  : {desc} (Severity {args.severity})")
    print(f"  Modes       : {args.modes}")
    print("=" * 70)

    # --- Data ---
    loader = get_loader(
        corruption_type=args.corruption,
        severity=args.severity,
        data_dir=args.data_dir,
        clean_dir=clean_dir,
        on_the_fly=args.on_the_fly,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Dataset size: {len(loader.dataset)}")

    for mode in args.modes:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        model = load_model(args.model, device)
        
        wrapper = None
        if mode == 'normal':
            acc = eval_normal(model, loader, device)
        elif mode == 'tent':
            wrapper = setup_tent(model, lr=args.lr, steps=args.steps)
            acc = eval_tta(wrapper, loader, device)
        elif mode == 'eata':
            fisher_loader = get_loader(
                args.corruption, args.severity, args.data_dir, clean_dir,
                args.on_the_fly, batch_size=32, num_workers=args.num_workers)
            fishers = compute_fishers(model, fisher_loader, device, num_samples=500)
            wrapper = setup_eata(model, fishers=fishers, lr=args.lr, steps=args.steps)
            acc = eval_tta(wrapper, loader, device)
        elif mode == 'proto_tta':
            wrapper = setup_proto_tta(
                model, lr=args.proto_lr, steps=args.steps,
                use_importance=not args.proto_no_importance, use_confidence=True,
                adapt_all_prototypes=args.proto_all_prototypes,
                use_geometric_filter=True, geo_filter_threshold=args.proto_threshold,
                consensus_strategy='max',
                adaptation_mode='layernorm_attn_bias',
                use_branch_agreement=args.proto_branch_agreement,
                prototype_branch=args.proto_branch,
                similarity_mapping=args.proto_mapping,
                sigmoid_center=args.proto_sigmoid_center,
                sigmoid_temp=args.proto_sigmoid_temp,
                proto_weight=args.proto_weight if args.proto_plus else 1.0,
                logit_weight=args.logit_weight if args.proto_plus else 0.0,
            )
            acc = eval_tta(wrapper, loader, device)
        else:
            print(f"  Unknown mode: {mode}")
            continue

        print(f"  Top-1 Accuracy : {acc*100:.2f}%")
        
        # Report extra TTA metrics if available (like in ProtoViT)
        if wrapper and hasattr(wrapper, 'adaptation_stats'):
            stats = wrapper.adaptation_stats
            total = stats.get('total_samples', 0)
            adapted = stats.get('adapted_samples', 0)
            if total > 0:
                filter_rate = (total - adapted) / total * 100
                print(f"  Reliable Filter Rate : {filter_rate:.1f}% ({adapted}/{total} adapted)")
                print(f"  Total Scale Updates  : {stats.get('total_updates', 0)}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
