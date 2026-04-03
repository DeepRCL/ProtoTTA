#!/usr/bin/env python3
"""SAR adaptation for ProtoPFormer."""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def cancel_first_step(self):
        """Restore weights after ``first_step()`` without ``base_optimizer.step()``.

        If the second SAR pass has no reliable samples, we must not leave parameters
        stuck at ``w + e(w)`` — that corrupts all following batches (this was the
        main cause of SAR collapse on ProtoPFormer).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
        self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def update_ema(ema, new_data):
    return new_data if ema is None else 0.9 * ema + 0.1 * new_data


def collect_params(model):
    """Match ProtoViT SAR: only adapt selected normalization parameters.

    SAR is more brittle than Tent/EATA, so we keep its parameter selection
    conservative and avoid the top transformer blocks plus the final norm.
    """
    params, names = [], []
    for nm, m in model.named_modules():
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_name, p in m.named_parameters():
                if np_name in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np_name}")
    return params, names


def configure_model(model):
    """Configure model for SAR, following the ProtoViT implementation."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class SAR(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=None, reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        num_classes = getattr(model, 'num_classes', 1000)
        # ProtoPFormer logits are noticeably less peaked than ProtoViT's in our
        # Dogs-C runs; using the original 0.4*log(C) makes SAR reject almost all
        # samples (~0.4% selection rate). We therefore use a slightly more
        # permissive default margin here so SAR actually adapts on this backbone.
        self.margin_e0 = margin_e0 if margin_e0 is not None else 0.6 * math.log(num_classes)
        self.reset_constant_em = reset_constant_em
        self.ema = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)
        self.adaptation_stats = {'total_samples': 0, 'adapted_samples': 0, 'total_updates': 0}

    def forward(self, x):
        if self.episodic:
            self.reset()
        self.adaptation_stats['total_samples'] += x.size(0)
        for _ in range(self.steps):
            outputs, ema, reset_flag, num_adapted = forward_and_adapt_sar(
                x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema
            )
            self.ema = ema
            self.adaptation_stats['adapted_samples'] += num_adapted
            if num_adapted > 0:
                self.adaptation_stats['total_updates'] += num_adapted
            if reset_flag:
                self.reset()
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.ema = None

    def forward_no_adapt(self, x):
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.enable_grad()
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
    """Match ProtoViT SAR logic; always completes or fully cancels the SAM two-step."""
    optimizer.zero_grad()
    outputs = model(x)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs

    entropies = softmax_entropy(logits)
    reliable_idx = torch.where(entropies < margin)
    num_adapted_first = len(reliable_idx[0])
    if num_adapted_first == 0:
        return outputs, ema, False, 0

    loss = entropies[reliable_idx].mean(0)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    outputs_second = model(x)
    logits_second = outputs_second[0] if isinstance(outputs_second, tuple) else outputs_second
    # Same subsample as ProtoViT: re-filter entropy at perturbed weights
    entropies_second = softmax_entropy(logits_second)[reliable_idx]
    reliable_idx_second = torch.where(entropies_second < margin)
    num_adapted = len(reliable_idx_second[0])
    if num_adapted == 0:
        # Critical: undo perturbation — do not leave weights at w + e(w)
        optimizer.cancel_first_step()
        return outputs, ema, False, 0

    loss_second = entropies_second[reliable_idx_second].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    reset_flag = ema is not None and ema < reset_constant
    return outputs, ema, reset_flag, num_adapted


def setup_sar(
    model,
    lr=1e-4,
    steps=1,
    episodic=False,
    margin_e0=None,
    reset_constant_em=0.2,
    rho=0.05,
):
    """SAR with SAM.

    For ProtoPFormer we use a more permissive default margin than the original
    0.4*log(C), because the stricter threshold led to near-zero adaptation on
    Stanford Dogs-C.
    """
    model = configure_model(model)
    params, _ = collect_params(model)
    nc = getattr(model, "num_classes", None)
    if margin_e0 is None:
        margin_e0 = 0.6 * math.log(float(nc)) if nc is not None else 0.6 * math.log(1000.0)
    # Slightly gentler default LR for ViT LayerNorms than 1e-3 (often too aggressive)
    optimizer = SAM(params, torch.optim.SGD, rho=rho, lr=lr, momentum=0.9)
    return SAR(
        model,
        optimizer,
        steps=steps,
        episodic=episodic,
        margin_e0=margin_e0,
        reset_constant_em=reset_constant_em,
    )
