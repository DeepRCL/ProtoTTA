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
    if "vit" in mode:
        vit = getattr(model.image_encoder, "model", None)
        if vit is None:
            raise ValueError(
                "adapt_mode includes 'vit' but image_encoder has no .model "
                "(expected DinoFeaturizer)."
            )
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
    if "vit" in mode:
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
# ProtoTTA for ProtoS-ViT  (proto_entropy_v3 analogue)
#
# Key differences from ProtoViT:
#   • ProtoS-ViT uses cosine sim → softmax → depthwise conv → LayerNorm → relu
#     so similarity_score ∈ [0, ∞).  We apply sigmoid to get ∈ (0, 1).
#   • The head is NonNegLinear, so head_weight[c, :] ≥ 0 for all c, j.
#     A prototype is "target" for class c iff head_weight[c, j] > 0.
#   • We minimize binary entropy H(sigmoid(sim_j)) for target prototypes
#     → sharpens each prototype activation toward confident 0 or 1.
# ══════════════════════════════════════════════════════════════════════════════

class ProtoTTA(nn.Module):
    def __init__(self, model, optimizer, steps: int = 1,
                 use_importance: bool = True,
                 use_confidence: bool = True,
                 geo_filter_threshold: float = 0.3,
                 proto_weight: float = 1.0,
                 logit_weight: float = 0.0,
                 adapt_all_prototypes: bool = False):
        """
        Parameters
        ----------
        use_importance        Weight entropy loss by class-head weight (prototype importance).
        use_confidence        Weight loss by pseudo-label softmax confidence.
        geo_filter_threshold  Skip samples with weak raw prototype support.
        proto_weight          Weight of prototype binary-entropy loss (1.0 for ProtoTTA).
        logit_weight          Weight of logit entropy (>0 activates ProtoTTA+).
        adapt_all_prototypes  If True, treat every prototype as a target (no class filter).
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.use_importance = use_importance
        self.use_confidence = use_confidence
        self.geo_filter_threshold = geo_filter_threshold
        self.proto_weight = proto_weight
        self.logit_weight = logit_weight
        self.adapt_all_prototypes = adapt_all_prototypes
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
        logits = out["pred"]  # [B, C]
        sim_score = out["similarity_score"]  # [B, P] relu output >= 0
        filter_score = out.get("proto_filter_score", sim_score)  # [B, P]

        eps = 1e-6

        with torch.no_grad():
            pred_class = logits.argmax(dim=1)  # [B]

            # ── Geometric filter: threshold the raw prototype support score ──
            if self.geo_filter_threshold > 0:
                max_sim = filter_score.max(dim=1)[0]  # [B]
                reliable = (max_sim > self.geo_filter_threshold).float()
            else:
                reliable = torch.ones(x.size(0), device=x.device)

            n_reliable = int(reliable.sum().item())
            self.adaptation_stats["adapted_samples"] += n_reliable
            if n_reliable == 0:
                return out

            # ── Target prototype mask ────────────────────────────────────────
            # classification_head.weight : [C, P]  (NonNeg → all entries ≥ 0)
            head_w = self.model.classification_head.weight.detach()  # [C, P]
            if self.adapt_all_prototypes:
                target_mask = torch.ones(x.size(0), sim_score.size(1), device=x.device)
            else:
                target_mask = (head_w[pred_class] > 0).float()  # [B, P]

        # ── Map similarity_score to Bernoulli prob via sigmoid ───────────────
        proto_probs = torch.sigmoid(sim_score).clamp(eps, 1 - eps)  # [B, P]

        # ── Binary entropy for target prototypes ────────────────────────────
        bin_ent = -(proto_probs * torch.log(proto_probs)
                    + (1 - proto_probs) * torch.log(1 - proto_probs))  # [B, P]

        sample_w = reliable.unsqueeze(1)  # [B, 1]

        if self.use_importance:
            imp = head_w[pred_class].clamp(min=0) * target_mask          # [B, P]
            imp = imp / (imp.sum(dim=1, keepdim=True) + eps)
            loss_per_sample = (bin_ent * target_mask * imp * sample_w).sum(dim=1)
        else:
            n_tgt = target_mask.sum(dim=1).clamp(min=1)
            loss_per_sample = (
                (bin_ent * target_mask * sample_w).sum(dim=1) / n_tgt
            )

        if self.use_confidence:
            with torch.no_grad():
                conf = logits.softmax(1).max(1)[0]  # [B]
            loss_proto = (loss_per_sample * conf * reliable).sum() / (reliable.sum() + eps)
        else:
            loss_proto = (loss_per_sample * reliable).sum() / (reliable.sum() + eps)

        loss = self.proto_weight * loss_proto
        if self.logit_weight > 0:
            logit_ent = (softmax_entropy(logits) * reliable).sum() / (reliable.sum() + eps)
            loss = loss + self.logit_weight * logit_ent

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.adaptation_stats["total_updates"] += n_reliable
        return out

    def reset(self):
        load_state(self.model, self.optimizer, self.model_state, self.optimizer_state)


def setup_proto_tta(model, lr: float = 1e-3, steps: int = 1,
                    use_importance: bool = True, use_confidence: bool = True,
                    geo_filter_threshold: float = 0.3,
                    adapt_all_prototypes: bool = False,
                    proto_weight: float = 1.0,
                    logit_weight: float = 0.0,
                    adapt_mode: str = "vit_layernorm_conv_proto_project"):
    model = configure_model(model, adapt_mode)
    params, names = collect_params(model, adapt_mode)
    preview = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
    print(f"  [ProtoTTA] adapting {len(params)} param tensors ({preview})")
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return ProtoTTA(
        model, optimizer, steps=steps,
        use_importance=use_importance,
        use_confidence=use_confidence,
        geo_filter_threshold=geo_filter_threshold,
        proto_weight=proto_weight,
        logit_weight=logit_weight,
        adapt_all_prototypes=adapt_all_prototypes,
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
            proto_weight, logit_weight, adapt_mode, steps, device):
    loader = get_loader(corruption, severity, cars_c_dir, batch_size, num_workers)
    print(f"  {corruption} sev={severity}  |  {len(loader.dataset)} images")
    results = {}

    for mode in modes:
        model = load_model(ckpt_path, device)
        print(f"\n  ── {mode.upper()} ──")
        if mode == "normal":
            acc = eval_normal(model, loader, device)
        elif mode == "tent":
            wrapper = setup_tent(model, lr=lr, steps=steps, adapt_mode=adapt_mode)
            acc = eval_tta(wrapper, loader, device)
        elif mode == "eata":
            # Compute Fisher on a small subset first
            configure_model(model, adapt_mode)
            fishers = compute_fishers(
                model,
                get_loader(corruption, severity, cars_c_dir, batch_size=32, num_workers=num_workers),
                device,
                adapt_mode=adapt_mode,
            )
            wrapper = setup_eata(model, lr=lr, steps=steps, adapt_mode=adapt_mode, fishers=fishers)
            acc = eval_tta(wrapper, loader, device)
        elif mode == "sar":
            wrapper = setup_sar(model, lr=lr, steps=steps, adapt_mode=adapt_mode)
            acc = eval_tta(wrapper, loader, device)
        elif mode == "proto_tta":
            wrapper = setup_proto_tta(
                model, lr=proto_lr, steps=steps,
                use_importance=True, use_confidence=True,
                geo_filter_threshold=proto_threshold,
                proto_weight=1.0, logit_weight=0.0,
                adapt_mode=adapt_mode,
            )
            acc = eval_tta(wrapper, loader, device)
        elif mode == "proto_tta_plus":
            wrapper = setup_proto_tta(
                model, lr=proto_lr, steps=steps,
                use_importance=True, use_confidence=True,
                geo_filter_threshold=proto_threshold,
                proto_weight=proto_weight,
                logit_weight=logit_weight,
                adapt_mode=adapt_mode,
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
                   help="LR for Tent / EATA / SAR (try 1e-4 if training is unstable)")
    p.add_argument("--proto_lr", type=float, default=3e-4,
                   help="LR for ProtoTTA / ProtoTTA+")
    p.add_argument("--proto_threshold", type=float, default=0.3,
                   help="Geometric filter threshold on raw prototype support (max_patch softmax similarity)")
    p.add_argument("--proto_weight", type=float, default=0.7,
                   help="Weight of prototype loss in ProtoTTA+ (remainder goes to logit entropy)")
    p.add_argument("--logit_weight", type=float, default=0.3,
                   help="Weight of logit entropy in ProtoTTA+")
    p.add_argument(
        "--adapt_mode",
        default="vit_layernorm_conv_proto_project",
        choices=[
            "vit",
            "layernorm",
            "layernorm_conv",
            "layernorm_proto",
            "layernorm_conv_proto",
            "layernorm_conv_proto_project",
            "vit_layernorm_conv_proto_project",
            "full_head",
        ],
        help=(
            "Substring tags: vit=DINO ViT LayerNorm+attn biases (Tent/EATA-style); "
            "conv/proto/project/head=prototype head. Default includes vit+full head "
            "(no classifier). full_head adds NonNegLinear; use smaller --lr."
        ),
    )
    p.add_argument("--steps", type=int, default=1,
                   help="Gradient steps per batch")
    p.add_argument("--output", default=None,
                   help="Path to save JSON results (optional)")
    args = p.parse_args()

    if args.adapt_mode == "full_head":
        args.adapt_mode = "vit_layernorm_conv_proto_project_head"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  Cars-C dir  : {args.cars_c_dir}")
    print(f"  Modes       : {args.modes}")
    print(f"  adapt_mode  : {args.adapt_mode}")
    print(f"  Device      : {device}")
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
                    args.proto_weight, args.logit_weight,
                    args.adapt_mode, args.steps, device,
                )
                all_results[key] = res
    else:
        key = f"{args.corruption}_sev{args.severity}"
        all_results[key] = run_one(
            args.ckpt, args.corruption, args.severity, args.modes,
            args.cars_c_dir, args.batch_size, args.num_workers,
            args.lr, args.proto_lr, args.proto_threshold,
            args.proto_weight, args.logit_weight,
            args.adapt_mode, args.steps, device,
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
