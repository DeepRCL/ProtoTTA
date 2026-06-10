#!/usr/bin/env python3
"""
run_inference_cars_c.py
========================
Test-time adaptation inference for ProtoS-ViT on Stanford Cars-C.

Supported methods: normal, tent, eata, sar, proto_tta, proto_tta_plus

ProtoTTA design for ProtoS-ViT (proto_entropy_v3 analogue)
-----------------------------------------------------------
ProtoS-ViT forward path produces:
  similarity_score   : [B, P]  ← relu(LayerNorm(depthwise_conv(softmax_cos_sim)))
                                 values in [0, ∞)
  proto_filter_score : [B, P]  ← max_patch(softmax_cos_sim)
                                 values in [0, 1]
  pred               : [B, C]  ← NonNegLinear(similarity_score)

ProtoTTA objective:
  1. Use proto_filter_score for geometric filtering (same role as ProtoViT's
     raw prototype similarity threshold)
  2. Pseudo-label c = argmax(logits)
  3. Target mask: prototype j is "target" iff head_weight[c, j] > 0 (NonNeg head)
  4. Map similarity_score → [0,1] via sigmoid for the prototype entropy loss
  5. Minimize binary entropy H(p) = -(p log p + (1-p) log(1-p)) for target prototypes
     → sharpens each prototype activation to confidently 0 or 1
  6. Optionally weight by importance (head_weight[c, :]) and prediction confidence
  7. ProtoTTA+: adds logit entropy to the loss with a separate weight

Usage
-----
# Single corruption
python run_inference_cars_c.py \\
    --ckpt logs/.../checkpoints/epoch_076.ckpt \\
    --cars_c_dir /home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars_c \\
    --modes normal tent eata sar proto_tta proto_tta_plus \\
    --corruption gaussian_noise --severity 5

# Full sweep
python run_inference_cars_c.py \\
    --ckpt logs/.../checkpoints/epoch_076.ckpt \\
    --cars_c_dir /home/mahdi.abootorabi/protovit/InfoDisent/Classificators/datasets/cars_c \\
    --modes normal proto_tta proto_tta_plus \\
    --all_corruptions \\
    --output results/cars_c_tta.json
"""
import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from einops import rearrange

# ── Project root so that `src.*` imports work regardless of CWD ───────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Needed for checkpoint unpickling paths like `dinov2.models.*`
sys.path.insert(0, str(ROOT / "src" / "learning" / "models" / "backbones"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import pyrootutils
pyrootutils.setup_root(ROOT, indicator=".project-root", pythonpath=True)

from src.shared_utils.torch_safe_load import register_trusted_checkpoint_globals
from src.learning.models.ClassificationModulePrune import ClassificationModulePrototype
from src.learning.models.utils.modules import LayerNorm as ProtoLayerNorm
from src.learning.models.backbones.dinov2.layers.attention import Attention

register_trusted_checkpoint_globals()

# Both nn.LayerNorm (inside DINOv2 blocks) and the custom ProtoLayerNorm
# (used in the similarity aggregation head) are adaptable norm layers.
ADAPTABLE_NORM_TYPES = (nn.LayerNorm, ProtoLayerNorm)


def _is_vit_attention_module(module: nn.Module) -> bool:
    """Match DINOv2 attention blocks robustly across runtime/checkpoint variants."""
    if isinstance(module, Attention):
        return True
    return (
        hasattr(module, "qkv")
        and hasattr(module, "proj")
        and isinstance(getattr(module, "qkv"), nn.Linear)
        and isinstance(getattr(module, "proj"), nn.Linear)
    )


def _collect_vit_tta_params(vit: nn.Module, prefix: str = "image_encoder.model"):
    """LayerNorm γ,β + attention qkv/proj biases (Tent/EATA-style ViT TTA)."""
    params, names = [], []
    for nm, m in vit.named_modules():
        path = f"{prefix}.{nm}" if nm else prefix
        if isinstance(m, nn.LayerNorm):
            params.append(m.weight)
            names.append(f"{path}.weight")
            if m.bias is not None:
                params.append(m.bias)
                names.append(f"{path}.bias")
        if _is_vit_attention_module(m):
            if m.qkv.bias is not None:
                params.append(m.qkv.bias)
                names.append(f"{path}.qkv.bias")
            if m.proj.bias is not None:
                params.append(m.proj.bias)
                names.append(f"{path}.proj.bias")
    return params, names


def _configure_vit_tta(vit: nn.Module) -> None:
    for m in vit.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
        if _is_vit_attention_module(m):
            if m.qkv.bias is not None:
                m.qkv.bias.requires_grad_(True)
            if m.proj.bias is not None:
                m.proj.bias.requires_grad_(True)


def _collect_vit_ln_only_params(vit: nn.Module, prefix: str = "image_encoder.model"):
    """LayerNorm γ,β only — matches the original Tent / EATA / SAR papers."""
    params, names = [], []
    for nm, m in vit.named_modules():
        path = f"{prefix}.{nm}" if nm else prefix
        if isinstance(m, nn.LayerNorm):
            params.append(m.weight)
            names.append(f"{path}.weight")
            if m.bias is not None:
                params.append(m.bias)
                names.append(f"{path}.bias")
    return params, names


def _configure_vit_ln_only(vit: nn.Module) -> None:
    """Enable grad only for LayerNorm in the ViT (no attention biases)."""
    for m in vit.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CORRUPTION_TYPES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise",
    "impulse_noise", "jpeg_compression", "motion_blur",
    "pixelate", "shot_noise", "spatter", "speckle_noise",
]

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device) -> ClassificationModulePrototype:
    register_trusted_checkpoint_globals()
    model = ClassificationModulePrototype.load_from_checkpoint(
        ckpt_path, map_location=device
    )
    model = model.to(device).eval()
    # Silence Lightning's self.log() calls when running outside a Trainer.
    # During training the Trainer handles this; here we just discard the metric.
    model.log = lambda *args, **kwargs: None
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def get_loader(corruption: str, severity: int, cars_c_dir: str,
               batch_size: int = 64, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    src = Path(cars_c_dir) / corruption / str(severity)
    if not src.exists():
        raise FileNotFoundError(f"Cars-C split not found: {src}")
    dataset = datasets.ImageFolder(str(src), transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _logits(out):
    """Extract logits from a ProtoS-ViT output dict or plain tensor."""
    if isinstance(out, dict):
        return out["pred"]
    return out


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def copy_state(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_state(model, optimizer, ms, os_):
    model.load_state_dict(ms, strict=True)
    optimizer.load_state_dict(os_)


# ══════════════════════════════════════════════════════════════════════════════
# Parameter selection (ViT: LN + attn biases; ProtoS-ViT head: optional extras)
# ══════════════════════════════════════════════════════════════════════════════

def collect_params(model, mode: str = "layernorm"):
    """
    mode is a tag string; substrings enable components (combine as needed):

    * ``vit`` – DINOv2 inside ``image_encoder.model``: all ``nn.LayerNorm`` γ,β
      and attention ``qkv`` / ``proj`` **biases** only (standard Tent/EATA practice
      for ViTs; encoder stays in ``eval()`` so dropout stays off).
    * (always) ProtoLayerNorm / nn.LayerNorm in the **similarity head** (not under
      ``image_encoder``), same as before.
    * ``conv``, ``proto``, ``project``, ``head`` – prototype pipeline + classifier
      (see below).

    Presets include e.g. ``vit_layernorm_conv_proto_project`` (backbone TTA + head).
    """
    params, names = [], []
    if "vit_ln_only" in mode:
        vit = getattr(model.image_encoder, "model", None)
        if vit is None:
            raise ValueError("adapt_mode includes 'vit_ln_only' but image_encoder has no .model")
        vp, vn = _collect_vit_ln_only_params(vit)
        params.extend(vp)
        names.extend(vn)
    elif "vit" in mode:
        vit = getattr(model.image_encoder, "model", None)
        if vit is None:
            raise ValueError("adapt_mode includes 'vit' but image_encoder has no .model")
        vp, vn = _collect_vit_tta_params(vit)
        params.extend(vp)
        names.extend(vn)
    for nm, m in model.named_modules():
        if nm.startswith("image_encoder"):
            continue
        if isinstance(m, ADAPTABLE_NORM_TYPES):
            for np_, p in m.named_parameters():
                if np_ in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    if "conv" in mode:
        for nm, m in model.named_modules():
            if nm.startswith("image_encoder"):
                continue
            if isinstance(m, nn.Conv2d):
                for np_, p in m.named_parameters():
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    if "proto" in mode:
        params.append(model.prototype_embeddings.weight)
        names.append("prototype_embeddings.weight")
    if "project" in mode and getattr(model, "embed_projection", False):
        if hasattr(model, "project_head"):
            for np_, p in model.project_head.named_parameters():
                params.append(p)
                names.append(f"project_head.{np_}")
    if "head" in mode and hasattr(model, "classification_head"):
        for np_, p in model.classification_head.named_parameters():
            params.append(p)
            names.append(f"classification_head.{np_}")
    return params, names


def configure_model(model, mode: str = "layernorm"):
    """Select adaptable parameters; keep DINOv2 in eval() so attention dropout is off.

    Training used ``image_encoder.model.eval()`` with a frozen backbone. For TTA we
    still call ``image_encoder.eval()`` (and thus ``vit.eval()``), then turn
    ``requires_grad`` on only LayerNorm + attention biases in the ViT when
    ``vit`` is in ``mode`` — gradients flow in eval mode; this matches common
    Tent/EATA ViT recipes (no BatchNorm running-stats issue; LN is affine-only).
    """
    model.train()
    model.requires_grad_(False)
    model.image_encoder.eval()
    model.image_encoder.requires_grad_(False)
    if "vit_ln_only" in mode:
        vit = model.image_encoder.model
        _configure_vit_ln_only(vit)
    elif "vit" in mode:
        vit = model.image_encoder.model
        _configure_vit_tta(vit)
    # Head (outside image_encoder): norms, convs, prototypes, etc.
    for nm, m in model.named_modules():
        if nm.startswith("image_encoder"):
            continue
        if isinstance(m, ADAPTABLE_NORM_TYPES):
            m.requires_grad_(True)
    if "conv" in mode:
        for nm, m in model.named_modules():
            if nm.startswith("image_encoder"):
                continue
            if isinstance(m, nn.Conv2d):
                m.requires_grad_(True)
    if "proto" in mode:
        model.prototype_embeddings.weight.requires_grad_(True)
    if "project" in mode and getattr(model, "embed_projection", False):
        if hasattr(model, "project_head"):
            model.project_head.requires_grad_(True)
    if "head" in mode and hasattr(model, "classification_head"):
        model.classification_head.requires_grad_(True)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Normal (no adaptation)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_normal(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += _logits(model(imgs)).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


# ══════════════════════════════════════════════════════════════════════════════
# Tent (entropy minimisation of LayerNorm params)
# ══════════════════════════════════════════════════════════════════════════════

class Tent(nn.Module):
    def __init__(self, model, optimizer, steps: int = 1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.model_state, self.optimizer_state = copy_state(model, optimizer)
        self.adaptation_stats = {"total_samples": 0, "adapted_samples": 0, "total_updates": 0}

    def forward(self, x):
        self.adaptation_stats["total_samples"] += x.size(0)
        self.adaptation_stats["adapted_samples"] += x.size(0)
        for _ in range(self.steps):
            out = self.model(x)
            loss = softmax_entropy(_logits(out)).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.adaptation_stats["total_updates"] += x.size(0)
        return out

    def reset(self):
        load_state(self.model, self.optimizer, self.model_state, self.optimizer_state)


def setup_tent(model, lr: float = 1e-3, steps: int = 1, adapt_mode: str = "layernorm"):
    model = configure_model(model, adapt_mode)
    params, names = collect_params(model, adapt_mode)
    preview = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
    print(f"  [Tent] adapting {len(params)} param tensors ({preview})")
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return Tent(model, optimizer, steps=steps)


# ══════════════════════════════════════════════════════════════════════════════
# EATA (Efficient Anti-forgetting TTA)
# ══════════════════════════════════════════════════════════════════════════════

class EATA(nn.Module):
    def __init__(self, model, optimizer, fishers=None, fisher_alpha: float = 2000.,
                 steps: int = 1, e_margin=None, d_margin: float = 0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.e_margin = e_margin if e_margin is not None else math.log(196) / 2 - 1
        self.d_margin = d_margin
        self.current_model_probs = None
        self.model_state, self.optimizer_state = copy_state(model, optimizer)
        self.adaptation_stats = {"total_samples": 0, "adapted_samples": 0, "total_updates": 0}

    def forward(self, x):
        self.adaptation_stats["total_samples"] += x.size(0)
        for _ in range(self.steps):
            out = self._adapt_step(x)
        return out

    @torch.enable_grad()
    def _adapt_step(self, x):
        out = self.model(x)
        logits = _logits(out)
        entropys = softmax_entropy(logits)
        ids1 = torch.where(entropys < self.e_margin)[0]
        entropys_f = entropys[ids1]
        probs = logits.softmax(1)

        if self.current_model_probs is not None and ids1.numel() > 0:
            cos = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0), probs[ids1], dim=1
            )
            ids2 = torch.where(cos.abs() < self.d_margin)[0]
            entropys_f = entropys_f[ids2]
            if ids2.numel() > 0:
                self.current_model_probs = (
                    0.9 * self.current_model_probs
                    + 0.1 * probs[ids1][ids2].mean(0).detach()
                )
        elif ids1.numel() > 0:
            self.current_model_probs = probs[ids1].mean(0).detach()

        if entropys_f.numel() == 0:
            return out

        coeff = 1 / (torch.exp(entropys_f.clone().detach() - self.e_margin))
        loss = (entropys_f * coeff).mean()

        if self.fishers is not None:
            ewc = sum(
                self.fisher_alpha * (self.fishers[n][0] * (p - self.fishers[n][1]) ** 2).sum()
                for n, p in self.model.named_parameters() if n in self.fishers
            )
            loss = loss + ewc

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.adaptation_stats["adapted_samples"] += entropys_f.numel()
        self.adaptation_stats["total_updates"] += entropys_f.numel()
        return out

    def reset(self):
        load_state(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.current_model_probs = None


@torch.no_grad()
def compute_fishers(model, loader, device, adapt_mode: str, num_samples: int = 500):
    """Estimate diagonal Fisher information on a small subset of the loader."""
    model.eval()
    configure_model(model, adapt_mode)
    fishers = {
        nm: [torch.zeros_like(p), p.detach().clone()]
        for nm, p in model.named_parameters() if p.requires_grad
    }
    seen = 0
    for imgs, _ in loader:
        if seen >= num_samples:
            break
        imgs = imgs.to(device)
        with torch.enable_grad():
            out = model(imgs)
            softmax_entropy(_logits(out)).mean().backward()
        for nm, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fishers[nm][0] += p.grad.data.clone().pow(2)
        model.zero_grad()
        seen += imgs.size(0)
    for nm in fishers:
        fishers[nm][0] /= max(seen, 1)
    return fishers


def setup_eata(model, lr: float = 1e-3, steps: int = 1, adapt_mode: str = "layernorm",
               fishers=None):
    model = configure_model(model, adapt_mode)
    params, names = collect_params(model, adapt_mode)
    preview = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
    print(f"  [EATA] adapting {len(params)} param tensors ({preview})")
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return EATA(model, optimizer, fishers=fishers, steps=steps)


# ══════════════════════════════════════════════════════════════════════════════
# SAR (Sharpness-Aware Reliable entropy minimisation)
# ══════════════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer (inner loop for SAR)."""

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grads = [
            p.grad.norm(p=2)
            for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if not grads:
            return
        norm = torch.norm(torch.stack(grads), p=2)
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for g in self.param_groups:
            for p in g["params"]:
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAR(nn.Module):
    def __init__(self, model, optimizer, steps: int = 1,
                 margin_e0=None, reset_constant: float = 0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.margin_e0 = margin_e0 if margin_e0 is not None else math.log(196) * 0.4
        self.reset_constant = reset_constant
        self.ema = None
        self.model_state, self.optimizer_state = copy_state(model, optimizer)
        self.adaptation_stats = {"total_samples": 0, "adapted_samples": 0, "total_updates": 0}

    def forward(self, x):
        self.adaptation_stats["total_samples"] += x.size(0)
        for _ in range(self.steps):
            out, self.ema, reset, n_adapted = self._adapt_step(x, self.ema)
            if reset:
                load_state(self.model, self.optimizer, self.model_state, self.optimizer_state)
                self.ema = None
            self.adaptation_stats["adapted_samples"] += n_adapted
            self.adaptation_stats["total_updates"] += n_adapted
        return out

    @torch.enable_grad()
    def _adapt_step(self, x, ema):
        self.optimizer.zero_grad()
        out = self.model(x)
        logits = _logits(out)
        entropys = softmax_entropy(logits)
        ids1 = torch.where(entropys < self.margin_e0)[0]
        if ids1.numel() == 0:
            return out, ema, False, 0

        entropys[ids1].mean().backward()
        self.optimizer.first_step(zero_grad=True)

        out2 = self.model(x)
        entropys2 = softmax_entropy(_logits(out2))[ids1]
        ids2 = torch.where(entropys2 < self.margin_e0)[0]
        loss2 = entropys2[ids2].mean() if ids2.numel() > 0 else torch.tensor(float("nan"))

        if not torch.isnan(loss2):
            ema = 0.9 * ema + 0.1 * loss2.item() if ema is not None else loss2.item()
            loss2.backward()
        self.optimizer.second_step(zero_grad=True)

        reset = ema is not None and ema < self.reset_constant
        return out, ema, reset, ids2.numel()

    def reset(self):
        load_state(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.ema = None


def setup_sar(model, lr: float = 1e-3, steps: int = 1, adapt_mode: str = "layernorm"):
    model = configure_model(model, adapt_mode)
    params, names = collect_params(model, adapt_mode)
    preview = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
    print(f"  [SAR] adapting {len(params)} param tensors ({preview})")
    optimizer = SAM(params, torch.optim.SGD, lr=lr, momentum=0.9)
    return SAR(model, optimizer, steps=steps)


# ══════════════════════════════════════════════════════════════════════════════
# ProtoTTA for ProtoS-ViT
#
# Two design choices that differ from ProtoViT:
#
# 1. ADAPT MODE: The updated parameter set is controlled by ``adapt_mode``.
#    In this project we default ProtoTTA / ProtoTTA+ to LayerNorm-only updates
#    to match Tent / EATA / SAR (no attention biases). Prototype embeddings are
#    NOT moved. The prototype entropy loss provides richer, class-specific
#    gradient signal to update those normalization params compared to Tent's
#    uniform logit entropy.
#
# 2. PROTOTYPE SELECTION (ProtoS-ViT-specific):
#    ProtoS-ViT does not expose a fixed prototype_class_identity like ProtoViT.
#    Its NonNegLinear head is dense and shared, so class-level positive weights
#    are far too broad. We therefore select targets from the per-sample predicted-
#    class importance scores that the model already computes at inference time.
#    We use similarity_score (the actual classifier input) mapped through
#    sigmoid, which keeps ProtoS-ViT target prototypes moving toward stronger
#    activation instead of letting entropy minimization drive them negative.
# ══════════════════════════════════════════════════════════════════════════════

class ProtoTTA(nn.Module):
    def __init__(self, model, optimizer, steps: int = 1,
                 use_confidence: bool = True,
                 geo_filter_threshold: float = 0.3,
                 conf_filter_threshold: float = 0.0,
                 agreement_filter_threshold: float = 0.0,
                 reliability_mode: str = "support",
                 active_proto_threshold: float = 0.1,
                 active_proto_min: int = 1,
                 active_proto_max: int = 8,
                 proto_weight: float = 1.0,
                 logit_weight: float = 0.0,
                 alpha_separation: float = 0.0,
                 hs_alpha: float = 0.01,
                 hs_gamma: float = 0.01,
                 usage_weight: float = 1.0,
                 proto_objective: str = "binary_entropy",
                 target_source: str = "class_importance",
                 target_topk: int = 0,
                 target_mass: float = 0.0,
                 target_rel_threshold: float = 0.1,
                 adaptive_blend: bool = False,
                 consistency_weight: float = 0.0,
                 conflict_aware: bool = False,
                 warmup_batches: int = 0,
                 spatial_sharpness_weight: float = 0.0,
                 spatial_temperature: float = 1.0,
                 reset_mode: str = "none",
                 reset_frequency: int = 10):
        """
        Parameters
        ----------
        use_confidence        Weight loss by pseudo-label softmax confidence.
        geo_filter_threshold  Skip samples whose strongest prototype activation
                              is below this threshold (unreliable images).
        conf_filter_threshold Skip samples whose prediction confidence is below
                              this threshold.
        agreement_filter_threshold  Require agreement between pre-head and
                              post-head prototype rankings. 0 disables it.
        reliability_mode      One of:
                              - support: legacy max-support gate
                              - sparsity: ProtoS-ViT-specific gate using the
                                number of active prototypes in similarity_score.
        active_proto_threshold Prototype activity threshold for the sparsity gate.
        active_proto_min      Minimum number of active prototypes to adapt on.
        active_proto_max      Maximum number of active prototypes to adapt on.
        proto_weight          Weight of prototype binary-entropy loss.
        logit_weight          Weight of logit entropy (>0 ⇒ ProtoTTA+).
        alpha_separation      Optional separation weight. Defaults to 0 for
                              ProtoS-ViT because target entropy is the main
                              objective used here.
        hs_alpha             Alpha for the training-style WeightedHs sparsity.
        hs_gamma             Gamma for the training-style WeightedHs sparsity.
        usage_weight         Weight for the training-time prototype-usage term.
        proto_objective       One of:
                              - binary_entropy: current activation-sharpening loss
                              - importance_entropy: sharpen predicted-class
                                prototype contribution distribution directly
                              - importance_hoyer: encourage sparse predicted-class
                                importance with a Hoyer-style sparsity penalty.
                              - patch_entropy: minimize patch-wise categorical
                                entropy across prototypes using the original
                                ProtoS-ViT softmax-over-prototypes signal.
                              - train_reg: mirror ProtoS-ViT training by using
                                WeightedHs sparsity on full importance plus the
                                prototype-usage regularizer l_t.
        target_source         One of:
                              - class_importance (existing behavior),
                              - shared_support (class-agnostic),
                              - hybrid (mix of both).
        target_topk           Keep only the top-k predicted-class prototypes per
                              sample. Helps suppress weak noisy tail prototypes.
        target_mass           If > 0, keep the smallest set of prototypes whose
                              normalized importance reaches this mass. Applied
                              before target_topk if enabled.
        target_rel_threshold  Keep prototypes whose importance is at least this
                              fraction of the sample's maximum importance.
                              Helps drop weak noisy tail prototypes.
        adaptive_blend        If True, blend prototype and logit losses per
                              sample using confidence as the gate.
        consistency_weight    Optional BCE alignment between sigmoid(similarity)
                              and proto_filter_score to reduce head/support drift.
        conflict_aware        If True and logit_weight > 0, down-weight proto loss
                              when proto and logit gradients conflict (negative cosine).
        warmup_batches        Number of initial batches to run logit-only in
                              ProtoTTA+ before enabling proto-specific losses.
        spatial_sharpness_weight
                              Optional weight for patch-level sharpness loss on
                              target prototypes using cosine_sim over patches.
        spatial_temperature   Temperature for softmax over patches.
        reset_mode           One of: none, episodic, periodic.
        reset_frequency      Reset every N batches when reset_mode=periodic.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.use_confidence = use_confidence
        self.geo_filter_threshold = geo_filter_threshold
        self.conf_filter_threshold = conf_filter_threshold
        self.agreement_filter_threshold = agreement_filter_threshold
        self.reliability_mode = reliability_mode
        self.active_proto_threshold = active_proto_threshold
        self.active_proto_min = active_proto_min
        self.active_proto_max = active_proto_max
        self.proto_weight = proto_weight
        self.logit_weight = logit_weight
        self.alpha_separation = alpha_separation
        self.hs_alpha = hs_alpha
        self.hs_gamma = hs_gamma
        self.usage_weight = usage_weight
        self.proto_objective = proto_objective
        self.target_source = target_source
        self.target_topk = target_topk
        self.target_mass = target_mass
        self.target_rel_threshold = target_rel_threshold
        self.adaptive_blend = adaptive_blend
        self.consistency_weight = consistency_weight
        self.conflict_aware = conflict_aware
        self.warmup_batches = warmup_batches
        self.spatial_sharpness_weight = spatial_sharpness_weight
        self.spatial_temperature = spatial_temperature
        self.reset_mode = reset_mode
        self.reset_frequency = reset_frequency
        self.model_state, self.optimizer_state = copy_state(model, optimizer)
        self.adaptation_stats = {"total_samples": 0, "adapted_samples": 0, "total_updates": 0}
        self.batch_count = 0

    def forward(self, x):
        if self.reset_mode == "episodic":
            self.reset_model()
        elif (
            self.reset_mode == "periodic"
            and self.batch_count > 0
            and self.batch_count % self.reset_frequency == 0
        ):
            self.reset_model()

        self.adaptation_stats["total_samples"] += x.size(0)
        for _ in range(self.steps):
            out = self._adapt_step(x)
        self.batch_count += 1
        return out

    @torch.enable_grad()
    def _adapt_step(self, x):
        out = self.model(x)
        logits = out["pred"]                   # [B, C]
        cosine_sim = out["cosine_sim"]        # [B, P, spatial], raw cosine in [-1, 1]
        similarity_score = out["similarity_score"]  # [B, P], classifier input in [0, inf)
        proto_filter_score = out["proto_filter_score"]  # [B, P], filtering signal in [0, 1]
        eps = 1e-6
        patch_proto_probs = (
            out["similarity_prototype"] + out["similarity_background"]
        ).clamp(min=eps, max=1.0)

        B, P, _ = cosine_sim.shape

        with torch.no_grad():
            # ProtoS-ViT already exposes a calibrated prototype-support signal.
            probs = logits.softmax(1)
            conf = probs.max(1)[0]  # [B]
            if self.reliability_mode == "sparsity":
                max_sim_per_sample = similarity_score.max(dim=1)[0]  # [B]
                active_proto_count = (similarity_score > self.active_proto_threshold).sum(dim=1).float()
                reliable = (active_proto_count >= float(self.active_proto_min)).float()
                if self.active_proto_max > 0:
                    reliable = reliable * (active_proto_count <= float(self.active_proto_max)).float()
                if self.geo_filter_threshold > 0:
                    reliable = reliable * (max_sim_per_sample > self.geo_filter_threshold).float()
            else:
                max_sim_per_sample = proto_filter_score.max(dim=1)[0]  # [B]
                if self.geo_filter_threshold > 0:
                    reliable = (max_sim_per_sample > self.geo_filter_threshold).float()
                else:
                    reliable = torch.ones(B, device=x.device)
            if self.conf_filter_threshold > 0:
                reliable = reliable * (conf > self.conf_filter_threshold).float()
            if self.agreement_filter_threshold > 0:
                proto_rank = proto_filter_score.argmax(dim=1)
                score_rank = similarity_score.argmax(dim=1)
                top1_agree = proto_rank.eq(score_rank).float()
                agreement = F.cosine_similarity(proto_filter_score, similarity_score, dim=1)
                reliable = reliable * (
                    (top1_agree > 0) | (agreement > self.agreement_filter_threshold)
                ).float()

            n_reliable = int(reliable.sum().item())
            self.adaptation_stats["adapted_samples"] += n_reliable
            if n_reliable == 0:
                return out

            pred_class = logits.argmax(dim=1)  # [B]
            head_w = self.model.classification_head.weight.detach().clamp_min(0.0)  # [C, P]
            class_weights = head_w[pred_class]  # [B, P]
            sample_importance = out["importance"][
                torch.arange(B, device=x.device), :, pred_class
            ].detach()  # [B, P]
            norm_similarity = similarity_score.detach() / (
                similarity_score.detach().max(dim=1, keepdim=True).values + eps
            )
            shared_support = (proto_filter_score.detach() * norm_similarity).clamp_min(0.0)
            if self.target_source == "class_importance":
                target_scores = sample_importance
            elif self.target_source == "shared_support":
                target_scores = shared_support
            elif self.target_source == "hybrid":
                norm_importance = sample_importance / (
                    sample_importance.max(dim=1, keepdim=True).values + eps
                )
                target_scores = 0.5 * (norm_importance + shared_support)
            else:
                raise ValueError(f"Unsupported target_source: {self.target_source}")
            target_mask = (target_scores > 0).float()  # [B, P]

            if self.target_rel_threshold > 0:
                max_score = target_scores.max(dim=1, keepdim=True).values
                rel_cutoff = self.target_rel_threshold * max_score
                target_mask = target_mask * (target_scores >= rel_cutoff).float()

            # Focus adaptation on the semantically dominant prototypes instead of
            # all weak tail contributors, which are especially noisy under blur.
            if self.target_mass > 0:
                norm_scores = target_scores / (target_scores.sum(dim=1, keepdim=True) + eps)
                sorted_scores, sorted_idx = torch.sort(norm_scores, dim=1, descending=True)
                cumulative = sorted_scores.cumsum(dim=1)
                keep_sorted = (cumulative <= self.target_mass).float()
                keep_sorted[:, 0] = 1.0
                first_over = (cumulative > self.target_mass)
                if first_over.any():
                    first_over_idx = first_over.float().argmax(dim=1, keepdim=True)
                    keep_sorted.scatter_(1, first_over_idx, 1.0)
                mass_mask = torch.zeros_like(target_mask)
                mass_mask.scatter_(1, sorted_idx, keep_sorted)
                target_mask = target_mask * mass_mask

            if self.target_topk > 0:
                k = min(self.target_topk, target_scores.shape[1])
                top_idx = torch.topk(target_scores, k=k, dim=1).indices
                topk_mask = torch.zeros_like(target_mask)
                topk_mask.scatter_(1, top_idx, 1.0)
                target_mask = target_mask * topk_mask

            # Rare fallback: if no prototype currently contributes to the predicted
            # class, keep the strongest class-specific prototype so adaptation
            # still has a target.
            if self.target_source == "shared_support":
                fallback_support = shared_support
            else:
                fallback_support = class_weights * out["similarity_score"].detach()

            # Per-sample fallbacks are applied only where the active importance set
            # is empty.
            # keep its strongest prototype so adaptation still has a target.
            empty_targets = target_mask.sum(dim=1) == 0
            if empty_targets.any():
                fallback_idx = fallback_support.argmax(dim=1, keepdim=True)
                target_mask = target_mask.clone()
                target_mask[empty_targets] = 0.0
                target_mask.scatter_(1, fallback_idx, 1.0)

            nontarget_mask = 1.0 - target_mask
            importance = target_scores * target_mask
            importance = importance / (importance.sum(dim=1, keepdim=True) + eps)

        sample_w = reliable.unsqueeze(1)  # [B, 1]
        logit_per_sample = softmax_entropy(logits)
        if self.proto_objective == "binary_entropy":
            # ProtoS-ViT's classifier already consumes similarity_score >= 0, so
            # sigmoid(.) turns entropy minimization into "sharpen toward 1" rather
            # than allowing target prototypes to collapse toward -1.
            proto_probs = torch.sigmoid(similarity_score).clamp(min=eps, max=1 - eps)
            bin_ent = -(proto_probs * torch.log(proto_probs)
                        + (1 - proto_probs) * torch.log(1 - proto_probs))  # [B, P]
            loss_proto = (bin_ent * importance * sample_w).sum(dim=1)
        elif self.proto_objective == "importance_entropy":
            class_importance_for_loss = out["importance"][
                torch.arange(B, device=x.device), :, pred_class
            ].clamp(min=0.0)
            active_importance = (class_importance_for_loss * target_mask).clamp(min=eps)
            norm_importance = active_importance / (active_importance.sum(dim=1, keepdim=True) + eps)
            importance_entropy = -(norm_importance * torch.log(norm_importance)).sum(dim=1)
            active_targets = target_mask.sum(dim=1)
            entropy_norm = torch.log(active_targets.clamp(min=2.0))
            entropy_norm = torch.where(active_targets > 1, entropy_norm, torch.ones_like(entropy_norm))
            loss_proto = importance_entropy / entropy_norm.clamp(min=1.0)
            proto_probs = torch.sigmoid(similarity_score).clamp(min=eps, max=1 - eps)
        elif self.proto_objective == "importance_hoyer":
            class_importance_for_loss = out["importance"][
                torch.arange(B, device=x.device), :, pred_class
            ].clamp(min=0.0)
            active_importance = class_importance_for_loss * target_mask
            l1 = active_importance.sum(dim=1)
            l2_sq = (active_importance ** 2).sum(dim=1)
            active_targets = target_mask.sum(dim=1).clamp(min=1.0)
            # This mirrors the training-time sparsity regularizer better than
            # entropy: minimize support size without forcing a brittle one-hot.
            loss_proto = (l1.square() / (l2_sq + eps)) / torch.sqrt(active_targets)
            proto_probs = torch.sigmoid(similarity_score).clamp(min=eps, max=1 - eps)
        elif self.proto_objective == "patch_entropy":
            # ProtoS-ViT's native signal is a softmax over prototypes per patch.
            # Adapting that categorical distribution is more faithful than
            # treating each prototype independently with a binary objective.
            global_proto_weights = self.model.classification_head.weight.detach().clamp_min(0.0).amax(dim=0)
            global_proto_weights = global_proto_weights / (global_proto_weights.sum() + eps)
            patch_entropy = -(
                patch_proto_probs * torch.log(patch_proto_probs) * global_proto_weights.view(1, P, 1)
            ).sum(dim=1)  # [B, spatial]
            loss_proto = patch_entropy.mean(dim=1)
            proto_probs = proto_filter_score.detach().clamp(min=eps, max=1 - eps)
        elif self.proto_objective == "train_reg":
            importance_all = out["importance"].clamp(min=0.0)  # [B, P, C]
            hs_loss = self.hs_alpha * (
                torch.norm(importance_all, p=1, dim=[1, 2]).square()
                / (torch.sum(importance_all.square() + eps, dim=[1, 2]))
            )
            hs_loss = hs_loss / (importance_all[0].numel() ** 0.5)
            hs_loss = hs_loss + self.hs_gamma * torch.norm(importance_all, p=2, dim=[1, 2])

            similarity_all = patch_proto_probs
            similarity_tmp = rearrange(similarity_all, "b p n -> (b n) p")
            usage_loss = -(torch.log(torch.tanh(torch.sum(similarity_tmp, dim=0)) + 1e-20).mean())
            # Training used classification loss + prototype regularizers. At TTA we
            # replace labels with detached pseudo-labels to preserve class semantics
            # while matching the source training objective as closely as possible.
            pseudo_ce = F.cross_entropy(logits, pred_class.detach(), reduction="none")
            loss_proto = hs_loss + pseudo_ce + self.usage_weight * usage_loss
            proto_probs = proto_filter_score.detach().clamp(min=eps, max=1 - eps)
        else:
            raise ValueError(f"Unsupported proto_objective: {self.proto_objective}")
        if self.consistency_weight > 0:
            cons_target = proto_filter_score.detach().clamp(min=eps, max=1 - eps)
            cons_per_proto = F.binary_cross_entropy(proto_probs, cons_target, reduction="none")
            loss_consistency = (cons_per_proto * sample_w).mean(dim=1)
        else:
            loss_consistency = torch.zeros_like(logit_per_sample)
        if self.spatial_sharpness_weight > 0:
            patch_probs = F.softmax(cosine_sim / max(self.spatial_temperature, eps), dim=2).clamp(min=eps, max=1 - eps)
            spatial_ent = -(patch_probs * torch.log(patch_probs)).sum(dim=2)  # [B, P]
            n_targets = target_mask.sum(dim=1).clamp(min=1.0)
            loss_spatial = (spatial_ent * target_mask * sample_w).sum(dim=1) / n_targets
        else:
            loss_spatial = torch.zeros_like(logit_per_sample)

        if self.adaptive_blend and self.logit_weight > 0:
            blend = conf.detach()
            loss = (
                self.proto_weight * blend * loss_proto
                + self.logit_weight * (1.0 - blend) * logit_per_sample
            )
            if self.consistency_weight > 0:
                loss = loss + self.consistency_weight * loss_consistency
            loss = (loss * reliable).sum() / (reliable.sum() + eps)
        elif self.use_confidence:
            loss_target = (loss_proto * conf * reliable).sum() / (reliable.sum() + eps)
        else:
            loss_target = (loss_proto * reliable).sum() / (reliable.sum() + eps)

        if not (self.adaptive_blend and self.logit_weight > 0):
            # Optional separation term. Kept off by default for ProtoS-ViT because
            # similarity_score is non-negative and target sharpening is the primary signal.
            loss_sep = torch.tensor(0.0, device=x.device)
            if self.alpha_separation > 0:
                nontarget_probs = torch.sigmoid(similarity_score).clamp(min=eps, max=1 - eps)
                sep     = -torch.log(1 - nontarget_probs) * nontarget_mask * sample_w
                n_ntgt  = nontarget_mask.sum(dim=1).clamp(min=1)
                loss_sep = (sep.sum(dim=1) / n_ntgt * reliable).sum() / (reliable.sum() + eps)

            loss_logit = (logit_per_sample * reliable).sum() / (reliable.sum() + eps)
            loss_proto_block = self.proto_weight * (loss_target + self.alpha_separation * loss_sep)
            if self.consistency_weight > 0:
                loss_proto_block = loss_proto_block + self.consistency_weight * (
                    (loss_consistency * reliable).sum() / (reliable.sum() + eps)
                )
            if self.spatial_sharpness_weight > 0:
                loss_proto_block = loss_proto_block + self.spatial_sharpness_weight * (
                    (loss_spatial * reliable).sum() / (reliable.sum() + eps)
                )

            # Optional stabilization: let logit-only adaptation shape the model
            # for the first few batches before injecting proto-specific gradients.
            if self.batch_count < self.warmup_batches:
                loss_proto_block = loss_proto_block * 0.0

            if self.logit_weight > 0:
                if self.conflict_aware and self.batch_count >= self.warmup_batches:
                    gate = self._grad_alignment_gate(loss_proto_block, loss_logit)
                    loss_proto_block = loss_proto_block * gate
                loss = loss_proto_block + self.logit_weight * loss_logit
            else:
                loss = loss_proto_block

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.adaptation_stats["total_updates"] += n_reliable
        return out

    def _grad_alignment_gate(self, proto_loss: torch.Tensor, logit_loss: torch.Tensor) -> torch.Tensor:
        """Return max(cos(g_proto, g_logit), 0) as a detached scalar gate."""
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
        if not params:
            return torch.tensor(1.0, device=proto_loss.device)

        g_proto = torch.autograd.grad(
            proto_loss, params, retain_graph=True, allow_unused=True
        )
        g_logit = torch.autograd.grad(
            logit_loss, params, retain_graph=True, allow_unused=True
        )
        dot = torch.tensor(0.0, device=proto_loss.device)
        n1 = torch.tensor(0.0, device=proto_loss.device)
        n2 = torch.tensor(0.0, device=proto_loss.device)
        valid = 0
        for gp, gl in zip(g_proto, g_logit):
            if gp is None or gl is None:
                continue
            dot = dot + (gp * gl).sum()
            n1 = n1 + (gp * gp).sum()
            n2 = n2 + (gl * gl).sum()
            valid += 1
        if valid == 0:
            return torch.tensor(1.0, device=proto_loss.device)
        cos = dot / (torch.sqrt(n1) * torch.sqrt(n2) + 1e-12)
        return cos.clamp(min=0.0, max=1.0).detach()

    def reset(self):
        load_state(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def reset_model(self):
        self.model.load_state_dict(self.model_state, strict=True)


def setup_proto_tta(model, lr: float = 1e-3, steps: int = 1,
                    use_confidence: bool = True,
                    geo_filter_threshold: float = 0.3,
                    conf_filter_threshold: float = 0.0,
                    agreement_filter_threshold: float = 0.0,
                    reliability_mode: str = "support",
                    active_proto_threshold: float = 0.1,
                    active_proto_min: int = 1,
                    active_proto_max: int = 8,
                    proto_weight: float = 1.0,
                    logit_weight: float = 0.0,
                    alpha_separation: float = 0.0,
                    hs_alpha: float = 0.01,
                    hs_gamma: float = 0.01,
                    usage_weight: float = 1.0,
                    proto_objective: str = "binary_entropy",
                    target_source: str = "class_importance",
                    target_topk: int = 0,
                    target_mass: float = 0.0,
                    target_rel_threshold: float = 0.1,
                    adaptive_blend: bool = False,
                    consistency_weight: float = 0.0,
                    conflict_aware: bool = False,
                    warmup_batches: int = 0,
                    spatial_sharpness_weight: float = 0.0,
                    spatial_temperature: float = 1.0,
                    reset_mode: str = "none",
                    reset_frequency: int = 10,
                    adapt_mode: str = "vit_ln_only"):
    """
    adapt_mode default is 'vit_ln_only': adapts LayerNorm weight+bias only
    in the ViT backbone plus LayerNorms outside ``image_encoder``. This keeps
    ProtoTTA aligned with the LN-only baseline updates. Prototype embeddings
    are not adapted.
    """
    model = configure_model(model, adapt_mode)
    params, names = collect_params(model, adapt_mode)
    preview = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
    print(f"  [ProtoTTA] adapting {len(params)} param tensors ({preview})")
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return ProtoTTA(
        model, optimizer, steps=steps,
        use_confidence=use_confidence,
        geo_filter_threshold=geo_filter_threshold,
        conf_filter_threshold=conf_filter_threshold,
        agreement_filter_threshold=agreement_filter_threshold,
        reliability_mode=reliability_mode,
        active_proto_threshold=active_proto_threshold,
        active_proto_min=active_proto_min,
        active_proto_max=active_proto_max,
        proto_weight=proto_weight,
        logit_weight=logit_weight,
        alpha_separation=alpha_separation,
        hs_alpha=hs_alpha,
        hs_gamma=hs_gamma,
        usage_weight=usage_weight,
        proto_objective=proto_objective,
        target_source=target_source,
        target_topk=target_topk,
        target_mass=target_mass,
        target_rel_threshold=target_rel_threshold,
        adaptive_blend=adaptive_blend,
        consistency_weight=consistency_weight,
        conflict_aware=conflict_aware,
        warmup_batches=warmup_batches,
        spatial_sharpness_weight=spatial_sharpness_weight,
        spatial_temperature=spatial_temperature,
        reset_mode=reset_mode,
        reset_frequency=reset_frequency,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop (shared for all TTA wrappers)
# ══════════════════════════════════════════════════════════════════════════════

def eval_tta(wrapper, loader, device):
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = wrapper(imgs)
        correct += _logits(out).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


# ══════════════════════════════════════════════════════════════════════════════
# Run one corruption / severity combination
# ══════════════════════════════════════════════════════════════════════════════

def run_one(ckpt_path, corruption, severity, modes, cars_c_dir,
            batch_size, num_workers, lr, proto_lr, proto_threshold,
            proto_weight, logit_weight, proto_conf_threshold,
            proto_agreement_threshold, proto_reliability_mode, proto_active_threshold,
            proto_active_min, proto_active_max, proto_hs_alpha, proto_hs_gamma, proto_usage_weight,
            proto_objective, proto_target_source, proto_consistency_weight,
            proto_conflict_aware, proto_warmup_batches,
            proto_spatial_sharpness_weight, proto_spatial_temperature,
            proto_target_topk, proto_target_mass, proto_target_rel_threshold, proto_adaptive_blend,
            proto_reset_mode, proto_reset_frequency,
            baseline_adapt_mode, proto_adapt_mode,
            steps, device):
    loader = get_loader(corruption, severity, cars_c_dir, batch_size, num_workers)
    print(f"  {corruption} sev={severity}  |  {len(loader.dataset)} images")
    results = {}

    for mode in modes:
        model = load_model(ckpt_path, device)
        print(f"\n  ── {mode.upper()} ──")
        if mode == "normal":
            acc = eval_normal(model, loader, device)
        elif mode == "tent":
            wrapper = setup_tent(model, lr=lr, steps=steps, adapt_mode=baseline_adapt_mode)
            acc = eval_tta(wrapper, loader, device)
        elif mode == "eata":
            configure_model(model, baseline_adapt_mode)
            fishers = compute_fishers(
                model,
                get_loader(corruption, severity, cars_c_dir, batch_size=32, num_workers=num_workers),
                device,
                adapt_mode=baseline_adapt_mode,
            )
            wrapper = setup_eata(model, lr=lr, steps=steps, adapt_mode=baseline_adapt_mode, fishers=fishers)
            acc = eval_tta(wrapper, loader, device)
        elif mode == "sar":
            wrapper = setup_sar(model, lr=lr, steps=steps, adapt_mode=baseline_adapt_mode)
            acc = eval_tta(wrapper, loader, device)
        elif mode == "proto_tta":
            wrapper = setup_proto_tta(
                model, lr=proto_lr, steps=steps,
                use_confidence=True,
                geo_filter_threshold=proto_threshold,
                conf_filter_threshold=proto_conf_threshold,
                agreement_filter_threshold=proto_agreement_threshold,
                reliability_mode=proto_reliability_mode,
                active_proto_threshold=proto_active_threshold,
                active_proto_min=proto_active_min,
                active_proto_max=proto_active_max,
                proto_weight=1.0, logit_weight=0.0,
                hs_alpha=proto_hs_alpha,
                hs_gamma=proto_hs_gamma,
                usage_weight=proto_usage_weight,
                proto_objective=proto_objective,
                target_source=proto_target_source,
                consistency_weight=proto_consistency_weight,
                conflict_aware=proto_conflict_aware,
                warmup_batches=proto_warmup_batches,
                spatial_sharpness_weight=proto_spatial_sharpness_weight,
                spatial_temperature=proto_spatial_temperature,
                target_topk=proto_target_topk,
                target_mass=proto_target_mass,
                target_rel_threshold=proto_target_rel_threshold,
                adaptive_blend=proto_adaptive_blend,
                reset_mode=proto_reset_mode,
                reset_frequency=proto_reset_frequency,
                adapt_mode=proto_adapt_mode,
            )
            acc = eval_tta(wrapper, loader, device)
        elif mode == "proto_tta_plus":
            wrapper = setup_proto_tta(
                model, lr=proto_lr, steps=steps,
                use_confidence=True,
                geo_filter_threshold=proto_threshold,
                conf_filter_threshold=proto_conf_threshold,
                agreement_filter_threshold=proto_agreement_threshold,
                reliability_mode=proto_reliability_mode,
                active_proto_threshold=proto_active_threshold,
                active_proto_min=proto_active_min,
                active_proto_max=proto_active_max,
                proto_weight=proto_weight,
                logit_weight=logit_weight,
                hs_alpha=proto_hs_alpha,
                hs_gamma=proto_hs_gamma,
                usage_weight=proto_usage_weight,
                proto_objective=proto_objective,
                target_source=proto_target_source,
                consistency_weight=proto_consistency_weight,
                conflict_aware=proto_conflict_aware,
                warmup_batches=proto_warmup_batches,
                spatial_sharpness_weight=proto_spatial_sharpness_weight,
                spatial_temperature=proto_spatial_temperature,
                target_topk=proto_target_topk,
                target_mass=proto_target_mass,
                target_rel_threshold=proto_target_rel_threshold,
                adaptive_blend=proto_adaptive_blend,
                reset_mode=proto_reset_mode,
                reset_frequency=proto_reset_frequency,
                adapt_mode=proto_adapt_mode,
            )
            acc = eval_tta(wrapper, loader, device)
        else:
            print(f"  Unknown mode: {mode}")
            continue

        acc_pct = round(acc * 100, 2)
        print(f"  Top-1 Accuracy: {acc_pct:.2f}%")
        results[mode] = acc_pct

        # Print per-sample adaptation stats when available
        wrapper_obj = locals().get("wrapper")
        if wrapper_obj is not None and hasattr(wrapper_obj, "adaptation_stats"):
            s = wrapper_obj.adaptation_stats
            total_s = s.get("total_samples", 0)
            adapted = s.get("adapted_samples", 0)
            if total_s > 0:
                rate = adapted / total_s * 100
                print(f"    Adapted {adapted}/{total_s} samples ({rate:.1f}%)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="TTA inference for ProtoS-ViT on Cars-C")
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint, e.g. logs/.../checkpoints/epoch_076.ckpt")
    p.add_argument("--cars_c_dir", required=True,
                   help="Root of cars_c dataset (contains corruption sub-folders)")
    p.add_argument("--modes", nargs="+",
                   default=["normal", "tent", "eata", "sar", "proto_tta", "proto_tta_plus"],
                   choices=["normal", "tent", "eata", "sar", "proto_tta", "proto_tta_plus"])
    p.add_argument("--corruption", default="gaussian_noise", choices=CORRUPTION_TYPES)
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--all_corruptions", action="store_true",
                   help="Sweep over every corruption type found in cars_c_dir")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="LR for Tent / EATA / SAR")
    p.add_argument("--proto_lr", type=float, default=None,
                   help="LR for ProtoTTA / ProtoTTA+. Defaults to --lr.")
    p.add_argument("--proto_threshold", type=float, default=0.3,
                   help="Geometric filter threshold on proto_filter_score support")
    p.add_argument("--proto_weight", type=float, default=0.7,
                   help="Weight of prototype loss in ProtoTTA+")
    p.add_argument("--logit_weight", type=float, default=0.3,
                   help="Weight of logit entropy in ProtoTTA+")
    p.add_argument("--proto_conf_threshold", type=float, default=0.1,
                   help="Require pseudo-label confidence above this threshold for ProtoTTA adaptation")
    p.add_argument("--proto_agreement_threshold", type=float, default=0.0,
                   help="Require agreement between proto_filter_score and similarity_score before adapting")
    p.add_argument("--proto_reliability_mode", type=str, default="support",
                   choices=["support", "sparsity"],
                   help="Reliability gate for ProtoTTA samples")
    p.add_argument("--proto_active_threshold", type=float, default=0.1,
                   help="Similarity-score threshold used by the sparsity reliability gate")
    p.add_argument("--proto_active_min", type=int, default=1,
                   help="Minimum number of active prototypes for sparsity reliability gate")
    p.add_argument("--proto_active_max", type=int, default=8,
                   help="Maximum number of active prototypes for sparsity reliability gate (0 disables upper bound)")
    p.add_argument("--proto_hs_alpha", type=float, default=0.01,
                   help="Alpha for training-style WeightedHs sparsity")
    p.add_argument("--proto_hs_gamma", type=float, default=0.01,
                   help="Gamma for training-style WeightedHs sparsity")
    p.add_argument("--proto_usage_weight", type=float, default=1.0,
                   help="Weight for the training-time prototype-usage regularizer")
    p.add_argument("--proto_objective", type=str, default="binary_entropy",
                   choices=["binary_entropy", "importance_entropy", "importance_hoyer", "patch_entropy", "train_reg"],
                   help="Proto-specific TTA objective for ProtoTTA/ProtoTTA+")
    p.add_argument("--proto_target_source", type=str, default="class_importance",
                   choices=["class_importance", "shared_support", "hybrid"],
                   help="How target prototypes are selected for ProtoTTA losses")
    p.add_argument("--proto_consistency_weight", type=float, default=0.0,
                   help="BCE consistency weight between sigmoid(similarity_score) and proto_filter_score")
    p.add_argument("--proto_conflict_aware", action="store_true",
                   help="Down-weight proto loss when proto/logit gradients conflict")
    p.add_argument("--proto_warmup_batches", type=int, default=0,
                   help="Run logit-only updates for the first N batches before proto losses")
    p.add_argument("--proto_spatial_sharpness_weight", type=float, default=0.0,
                   help="Weight of patch-level spatial entropy loss on target prototypes")
    p.add_argument("--proto_spatial_temperature", type=float, default=1.0,
                   help="Softmax temperature for patch-level spatial entropy")
    p.add_argument("--proto_target_topk", type=int, default=0,
                   help="Keep only the top-k predicted-class prototypes per sample for ProtoTTA losses")
    p.add_argument("--proto_target_mass", type=float, default=0.0,
                   help="Optional cumulative importance mass to keep before top-k pruning (0 disables)")
    p.add_argument("--proto_target_rel_threshold", type=float, default=0.1,
                   help="Keep prototypes with importance >= this fraction of sample max importance")
    p.add_argument("--proto_adaptive_blend", action="store_true",
                   help="Blend prototype and logit losses per sample using confidence as the gate")
    p.add_argument("--proto_reset_mode", choices=["none", "episodic", "periodic"], default="none",
                   help="Reset ProtoTTA parameters between batches")
    p.add_argument("--proto_reset_frequency", type=int, default=10,
                   help="Reset every N batches when --proto_reset_mode=periodic")
    # ── Adapt mode: separate for baselines vs ProtoTTA ──
    p.add_argument(
        "--baseline_adapt_mode",
        default="vit_ln_only",
        choices=[
            "vit",
            "vit_ln_only",
            "layernorm",
            "layernorm_conv",
            "layernorm_proto",
            "layernorm_conv_proto",
            "layernorm_conv_proto_project",
            "vit_layernorm_conv_proto_project",
        ],
        help=(
            "Adapt mode for Tent / EATA / SAR.  'vit_ln_only' adapts only "
            "LayerNorm weight+bias in the DINO-ViT backbone (matches the "
            "original papers).  'vit' also adds attention biases."
        ),
    )
    p.add_argument(
        "--proto_adapt_mode",
        default="vit_ln_only",
        choices=[
            "vit",
            "vit_conv",
            "vit_ln_only",
            "vit_ln_only_conv_proto",
            "layernorm",
            "layernorm_conv",
            "layernorm_proto",
            "layernorm_conv_proto",
            "layernorm_conv_proto_project",
            "vit_layernorm_conv_proto_project",
        ],
        help=(
            "Adapt mode for ProtoTTA / ProtoTTA+. Default 'vit_ln_only' "
            "keeps updates to LayerNorm parameters only (no attention biases), "
            "matching the baseline LN-only setup. "
            "'layernorm_conv_proto' is head-only (ViT frozen) — too weak alone."
        ),
    )
    p.add_argument("--steps", type=int, default=1,
                   help="Gradient steps per batch")
    p.add_argument("--output", default=None,
                   help="Path to save JSON results (optional)")
    args = p.parse_args()

    if args.proto_lr is None:
        args.proto_lr = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  Checkpoint       : {args.ckpt}")
    print(f"  Cars-C dir       : {args.cars_c_dir}")
    print(f"  Modes            : {args.modes}")
    print(f"  lr / proto_lr    : {args.lr} / {args.proto_lr}")
    print(f"  proto_threshold  : {args.proto_threshold}")
    print(f"  proto_conf_thr   : {args.proto_conf_threshold}")
    print(f"  proto_agree_thr  : {args.proto_agreement_threshold}")
    print(f"  proto_rel_mode   : {args.proto_reliability_mode}")
    print(f"  proto_active_thr : {args.proto_active_threshold}")
    print(f"  proto_active_rng : {args.proto_active_min}..{args.proto_active_max}")
    print(f"  proto_hs         : {args.proto_hs_alpha}/{args.proto_hs_gamma}")
    print(f"  proto_usage_w    : {args.proto_usage_weight}")
    print(f"  proto_objective  : {args.proto_objective}")
    print(f"  proto_target_src : {args.proto_target_source}")
    print(f"  proto_cons_w     : {args.proto_consistency_weight}")
    print(f"  proto_conflict   : {args.proto_conflict_aware}")
    print(f"  proto_warmup_b   : {args.proto_warmup_batches}")
    print(f"  proto_spatial_w  : {args.proto_spatial_sharpness_weight}")
    print(f"  proto_spatial_t  : {args.proto_spatial_temperature}")
    print(f"  proto_target_k   : {args.proto_target_topk}")
    print(f"  proto_target_m   : {args.proto_target_mass}")
    print(f"  proto_target_rel : {args.proto_target_rel_threshold}")
    print(f"  proto_adapt_blnd : {args.proto_adaptive_blend}")
    print(f"  proto_reset_mode : {args.proto_reset_mode}")
    print(f"  proto_reset_freq : {args.proto_reset_frequency}")
    print(f"  baseline_adapt   : {args.baseline_adapt_mode}")
    print(f"  proto_adapt      : {args.proto_adapt_mode}")
    print(f"  Device           : {device}")
    print("=" * 70)

    all_results = {}

    if args.all_corruptions:
        cars_c_root = Path(args.cars_c_dir)
        for corruption in CORRUPTION_TYPES:
            corr_dir = cars_c_root / corruption
            if not corr_dir.exists():
                print(f"\n  [SKIP] {corruption} not found in {cars_c_root}")
                continue
            severities = sorted(
                int(s.name) for s in corr_dir.iterdir()
                if s.is_dir() and s.name.isdigit()
            )
            for sev in severities:
                key = f"{corruption}_sev{sev}"
                print(f"\n{'=' * 70}\n  {key}\n{'=' * 70}")
                res = run_one(
                    args.ckpt, corruption, sev, args.modes,
                    args.cars_c_dir, args.batch_size, args.num_workers,
                    args.lr, args.proto_lr, args.proto_threshold,
                    args.proto_weight, args.logit_weight, args.proto_conf_threshold,
                    args.proto_agreement_threshold, args.proto_reliability_mode, args.proto_active_threshold,
                    args.proto_active_min, args.proto_active_max, args.proto_hs_alpha, args.proto_hs_gamma, args.proto_usage_weight,
                    args.proto_objective, args.proto_target_source, args.proto_consistency_weight,
                    args.proto_conflict_aware, args.proto_warmup_batches,
                    args.proto_spatial_sharpness_weight, args.proto_spatial_temperature,
                    args.proto_target_topk, args.proto_target_mass, args.proto_target_rel_threshold, args.proto_adaptive_blend,
                    args.proto_reset_mode, args.proto_reset_frequency,
                    args.baseline_adapt_mode, args.proto_adapt_mode,
                    args.steps, device,
                )
                all_results[key] = res
    else:
        key = f"{args.corruption}_sev{args.severity}"
        all_results[key] = run_one(
            args.ckpt, args.corruption, args.severity, args.modes,
            args.cars_c_dir, args.batch_size, args.num_workers,
            args.lr, args.proto_lr, args.proto_threshold,
            args.proto_weight, args.logit_weight, args.proto_conf_threshold,
            args.proto_agreement_threshold, args.proto_reliability_mode, args.proto_active_threshold,
            args.proto_active_min, args.proto_active_max, args.proto_hs_alpha, args.proto_hs_gamma, args.proto_usage_weight,
            args.proto_objective, args.proto_target_source, args.proto_consistency_weight,
            args.proto_conflict_aware, args.proto_warmup_batches,
            args.proto_spatial_sharpness_weight, args.proto_spatial_temperature,
            args.proto_target_topk, args.proto_target_mass, args.proto_target_rel_threshold, args.proto_adaptive_blend,
            args.proto_reset_mode, args.proto_reset_frequency,
            args.baseline_adapt_mode, args.proto_adapt_mode,
            args.steps, device,
        )

    # ── Summary table ──────────────────────────────────────────────────────
    col_w = 14
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY  (accuracy %)")
    print("=" * 70)
    header = f"  {'Corruption':<38}" + "".join(f"{m:>{col_w}}" for m in args.modes)
    print(header)
    print("-" * (38 + col_w * len(args.modes) + 2))
    for k, v in all_results.items():
        row = f"  {k:<38}" + "".join(
            f"{v.get(m, 'N/A'):>{col_w}}" for m in args.modes
        )
        print(row)
    print("=" * 70)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved → {args.output}")


if __name__ == "__main__":
    main()
