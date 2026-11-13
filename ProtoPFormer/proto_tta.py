"""
ProtoTTA for ProtoPFormer
=========================
Test-Time Adaptation using prototype-similarity entropy minimisation,
adapted from ProtoViT's proto_entropy.py to match ProtoPFormer's forward
signature, which returns:

    eval:  (logits, (cls_token_attn, distances, logits_global, logits_local))
    train: (logits, (student_token_attn, attn_loss, total_proto_act,
                      cls_attn_rollout, original_fea_len))

ProtoPFormer uses:
  - `model.prototype_class_identity`  → [num_prototypes, num_classes]
  - `model.last_layer`                → classification head (local branch)
  - LayerNorm layers and attention biases inside the ViT backbone for TTA

Implements ProtoTTA (fixed or sample-wise adaptive lambda) and the Tent and
EATA baselines of this backbone.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Utility helpers
# ============================================================================

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    opt_state   = deepcopy(optimizer.state_dict())
    return model_state, opt_state


def _forward_eval(model, x):
    """Run a forward pass in eval mode; always returns (logits, aux)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x)
    if was_training:
        model.train()
    return out


def _get_logits(out):
    """Extract logit tensor from ProtoPFormer's various output formats."""
    if isinstance(out, tuple):
        return out[0]
    return out


# ============================================================================
# collect_params / configure_model  (shared between all methods)
# ============================================================================

def collect_params(model, adaptation_mode='layernorm_only'):
    """Collect parameters to adapt (LayerNorms and attention biases)."""
    params, names = [], []

    if 'layernorm' in adaptation_mode:
        for nm, m in model.named_modules():
            # Check by type AND name (safest for timm/custom norms)
            classname = m.__class__.__name__
            is_norm = isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)) or \
                      'layernorm' in classname.lower() or 'batchnorm' in classname.lower()
            if is_norm:
                for np_, p in m.named_parameters():
                    if np_ in ('weight', 'bias'):
                        params.append(p)
                        names.append(f"{nm}.{np_}")

    if 'attn_bias' in adaptation_mode:
        for nm, m in model.named_modules():
            # Targets modules containing 'attn' or 'attention'
            if 'attn' in nm.lower() or 'attention' in nm.lower():
                for np_, p in m.named_parameters():
                    if 'bias' in np_:
                        params.append(p)
                        names.append(f"{nm}.{np_}")

    return params, names

def configure_model(model, adaptation_mode='layernorm_only', model_mode='train'):
    """Configure adaptation parameters; adaptation runs in training mode
    (as in the upstream Tent/EATA implementations)."""
    if model_mode != 'train':
        raise ValueError(f"Unknown adaptation model mode: {model_mode}")
    model.train()
    model.requires_grad_(False)

    params, names = collect_params(model, adaptation_mode)
    for p in params:
        p.requires_grad = True

    # Ensure BN layers use current batch stats (standard TTA practice)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


# ============================================================================
# Tent  (entropy minimisation over LayerNorms)
# ============================================================================

class Tent(nn.Module):
    """Standard Tent TTA — minimise Shannon entropy of model outputs."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(model, optimizer)
        self.adaptation_stats = {'total_samples': 0, 'adapted_samples': 0, 'total_updates': 0}

    def forward(self, x):
        if self.episodic:
            self.reset()
        self.adaptation_stats['total_samples'] += x.size(0)
        self.adaptation_stats['adapted_samples'] += x.size(0)
        for _ in range(self.steps):
            logits = self._forward_and_adapt(x)
            self.adaptation_stats['total_updates'] += 1
        return logits

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        out = self.model(x)
        logits = _get_logits(out)
        loss = softmax_entropy(logits).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return logits

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def forward_no_adapt(self, x):
        return _forward_eval(self.model, x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def softmax_entropy(logits):
    """Shannon entropy of softmax output — lower = more confident."""
    p = logits.softmax(dim=1)
    return -(p * torch.log(p + 1e-6)).sum(dim=1)


@torch.no_grad()
def relative_evidence_lambda(logits, branch_scores, branch_proto_identities,
                             topk=3, eps=1e-8):
    """Compare predicted-class prototype evidence with output evidence.

    Local and global ProtoPFormer prototypes share the same normalized score
    range, so they are concatenated before computing one top-k score per class.
    The returned coefficient is the prototype fraction in the unified loss.
    """
    available = [
        (scores, identities)
        for scores, identities in zip(branch_scores, branch_proto_identities)
        if scores is not None and identities is not None
    ]
    if not available:
        raise ValueError('relative_evidence requires prototype scores and class assignments')

    scores = torch.cat([item[0] for item in available], dim=1)
    identities = torch.cat([item[1] for item in available], dim=0)
    num_classes = logits.shape[1]
    class_scores = []
    for class_index in range(num_classes):
        class_mask = identities == class_index
        if not class_mask.any():
            class_scores.append(scores.new_full((scores.shape[0],), float('-inf')))
            continue
        per_class = scores[:, class_mask]
        k = min(max(int(topk), 1), per_class.shape[1])
        class_scores.append(per_class.topk(k, dim=1).values.mean(dim=1))
    class_scores = torch.stack(class_scores, dim=1)

    pred_class = logits.argmax(dim=1)
    selected = class_scores.gather(1, pred_class.unsqueeze(1)).squeeze(1)
    competitor = class_scores.scatter(
        1, pred_class.unsqueeze(1), float('-inf')
    ).max(dim=1).values
    proto_reliability = (
        (selected - competitor).clamp_min(0.0)
        / selected.abs().clamp_min(eps)
    ).clamp(0.0, 1.0)

    output_top2 = logits.softmax(dim=1).topk(2, dim=1).values
    output_reliability = (
        (output_top2[:, 0] - output_top2[:, 1])
        / output_top2[:, 0].clamp_min(eps)
    ).clamp(0.0, 1.0)
    coefficient = proto_reliability / (
        proto_reliability + output_reliability + eps
    )
    return coefficient.detach(), proto_reliability, output_reliability


def setup_tent(model, lr=1e-3, steps=1, episodic=False, model_mode='train'):
    model = configure_model(model, 'layernorm_only', model_mode=model_mode)
    params, _ = collect_params(model, 'layernorm_only')
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return Tent(model, optimizer, steps=steps, episodic=episodic)


# ============================================================================
# EATA  (efficient adaptive test-time adaptation)
# ============================================================================

def compute_fishers(model, loader, device, num_samples=500):
    """Compute Fisher information on a subset of loader for EATA."""
    model.eval()
    model.requires_grad_(True)

    fishers = {}
    total = 0

    for images, _ in loader:
        if total >= num_samples:
            break
        images = images.to(device)
        model.zero_grad()
        out = model(images)
        logits = _get_logits(out)
        # Use maximum-likelihood loss on pseudo labels
        loss = F.cross_entropy(logits, logits.argmax(dim=1))
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                if name not in fishers:
                    fishers[name] = p.grad.data.clone().pow(2)
                else:
                    fishers[name] += p.grad.data.clone().pow(2)

        total += images.size(0)

    # Normalise
    for name in fishers:
        fishers[name] /= total

    model.zero_grad()
    return fishers


class EATA(nn.Module):
    """EATA: Efficient Test-Time Adaptation with Fisher regularisation.

    Reference: Niu et al., ICML 2022.
    """

    def __init__(self, model, optimizer, fishers=None,
                 fisher_alpha=2000.0,
                 e_margin=0.4,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.e_margin = e_margin
        self.steps = steps
        self.episodic = episodic
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(model, optimizer)
        self.adaptation_stats = {'total_samples': 0, 'adapted_samples': 0, 'total_updates': 0}

    def forward(self, x):
        if self.episodic:
            self.reset()
        self.adaptation_stats['total_samples'] += x.size(0)
        for _ in range(self.steps):
            logits = self._forward_and_adapt(x)
        return logits

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        out = self.model(x)
        logits = _get_logits(out)

        # EATA filtering: only adapt samples with low entropy
        num_classes = logits.size(1)
        threshold = self.e_margin * torch.log(
            torch.tensor(num_classes, dtype=torch.float, device=logits.device))
        p = logits.softmax(dim=1)
        entropy = -(p * torch.log(p + 1e-6)).sum(dim=1)
        mask = (entropy < threshold).float()

        if mask.sum() == 0:
            return logits

        self.adaptation_stats['adapted_samples'] += int(mask.sum().item())
        self.adaptation_stats['total_updates'] += 1

        entropy_loss = (entropy * mask).sum() / (mask.sum() + 1e-8)

        # Fisher regularisation
        fisher_loss = torch.tensor(0., device=logits.device)
        if self.fishers is not None:
            for name, p in self.model.named_parameters():
                if p.requires_grad and name in self.fishers:
                    fisher_loss += (self.fishers[name].to(logits.device) *
                                    (p - p.detach()) ** 2).sum()

        loss = entropy_loss + self.fisher_alpha * fisher_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return logits

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def forward_no_adapt(self, x):
        return _forward_eval(self.model, x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_eata(model, fishers=None, lr=1e-3, steps=1, episodic=False,
               model_mode='train'):
    model = configure_model(model, 'layernorm_only', model_mode=model_mode)
    params, _ = collect_params(model, 'layernorm_only')
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return EATA(model, optimizer, fishers=fishers, steps=steps, episodic=episodic)


# ============================================================================
# ProtoTTA
# ============================================================================
# ProtoPFormer does NOT expose sub-prototype similarities like ProtoViT.
# Its prototype activations come from distance_2_similarity(distances), where
# `distances` is shape (B, num_prototypes, H, W).
# We use the global-max-pooled activations as "similarities".
# ============================================================================

class ProtoTTA(nn.Module):
    """ProtoTTA for ProtoPFormer — prototype-similarity entropy minimisation.

    Binary entropy of the predicted-class prototype activations of both
    branches, blended with output entropy, after geometric filtering of
    unreliable samples and with confidence weighting. The blend uses either a
    fixed lambda or the sample-wise adaptive controller.
    """

    def __init__(self, model, optimizer,
                 steps=1,
                 use_importance=True,
                 geo_filter_threshold=0.3,
                 prototype_branch='both',
                 similarity_mapping='sigmoid',
                 sigmoid_center=2.0,
                 sigmoid_temp=1.0,
                 proto_weight=1.0,
                 logit_weight=0.0,
                 shared_confidence_weighting=False,
                 gradient_normalize=False,
                 adaptive_lambda=False,
                 adaptive_lambda_strategy=None,
                 adaptive_delta0=0.25,
                 adaptive_topk=3,
                 router_min_consistency=0.25,
                 lambda_min=0.05,
                 lambda_max=0.95,
                 record_diagnostics=False,
                 record_vlm_evidence=False,
                 ):
        super().__init__()
        if prototype_branch != 'both':
            raise ValueError(f"Unsupported prototype_branch: {prototype_branch}")
        if similarity_mapping != 'sigmoid':
            raise ValueError(f"Unknown similarity_mapping: {similarity_mapping}")
        self.model = model
        self.optimizer = optimizer
        self.steps = steps

        self.use_importance = use_importance
        self.geo_filter_threshold = geo_filter_threshold
        self.sigmoid_center = sigmoid_center
        self.sigmoid_temp = sigmoid_temp
        self.proto_weight = proto_weight
        self.logit_weight = logit_weight
        self.shared_confidence_weighting = shared_confidence_weighting
        self.gradient_normalize = gradient_normalize
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_lambda_strategy = adaptive_lambda_strategy
        self.adaptive_delta0 = adaptive_delta0
        self.adaptive_topk = adaptive_topk
        self.router_min_consistency = router_min_consistency
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.record_diagnostics = record_diagnostics
        self.record_vlm_evidence = bool(record_vlm_evidence)
        self.last_vlm_evidence = None

        self.batch_count = 0

        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'total_updates': 0,
            'avg_reliability': [],
            'proto_loss': [],
            'output_loss': [],
            'proto_grad_norm': [],
            'output_grad_norm': [],
            'adaptive_lambda': [],
            'adaptive_lambda_raw': [],
            'proto_signal_reliability': [],
            'output_signal_reliability': [],
            'proto_gradient_consistency': [],
            'output_gradient_consistency': [],
            'adaptive_router_gate': [],
        }

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x):
        self.adaptation_stats['total_samples'] += x.size(0)

        for _ in range(self.steps):
            logits = self._forward_and_adapt(x)

        self.batch_count += 1
        return logits

    # -------------------------------------------------------------------------
    # Core adaptation step
    # -------------------------------------------------------------------------

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        # --- 1. Forward pass (train mode gives richer output) ---
        out = self.model(x)
        logits = _get_logits(out)

        local_raw, global_raw = self._get_proto_activations(out)
        if local_raw is None:
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.adaptation_stats['adapted_samples'] += x.size(0)
            self.adaptation_stats['total_updates'] += 1
            return logits

        local_scores = self._normalize_similarity(local_raw)
        global_scores = self._normalize_similarity(global_raw) if global_raw is not None else None
        if self.record_vlm_evidence:
            self.last_vlm_evidence = {
                'logits': logits.detach().float().cpu(),
                'local_raw': local_raw.detach().float().cpu(),
                'global_raw': (
                    None if global_raw is None else global_raw.detach().float().cpu()
                ),
            }

        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            local_proto_identities = None
            if local_scores is not None:
                local_pci = self.model.prototype_class_identity.to(logits.device)
                local_proto_identities = local_pci.argmax(dim=1)

            global_proto_identities = None
            if global_scores is not None and hasattr(self.model, 'prototype_class_identity_global'):
                global_pci = self.model.prototype_class_identity_global.to(logits.device)
                global_proto_identities = global_pci.argmax(dim=1)

        # --- 3. Geometric filter ---
        with torch.no_grad():
            if local_scores is not None:
                reliability_score = local_scores.max(dim=1)[0]
            else:
                reliability_score = global_scores.max(dim=1)[0]

            if local_scores is not None and global_scores is not None:
                local_max_sim = local_scores.max(dim=1)[0]
                global_max_sim = global_scores.max(dim=1)[0]
                global_coe = float(getattr(self.model, 'global_coe', 0.5))
                reliability_score = ((1.0 - global_coe) * local_max_sim +
                                     global_coe * global_max_sim)
            reliable_mask = (reliability_score > self.geo_filter_threshold)
            reliable_mask = reliable_mask.float()

            # Debug logging (every 50 batches)
            if self.batch_count % 50 == 0 and self.batch_count > 0:
                 print(f" [Batch {self.batch_count + 1}] Max Sim Avg: {reliability_score.mean():.3f}, Threshold: {self.geo_filter_threshold}")

            adapted = int(reliable_mask.sum().item())
            self.adaptation_stats['adapted_samples'] += adapted
            self.adaptation_stats['avg_reliability'].append(float(reliability_score.mean().item()))
            if reliable_mask.sum() == 0:
                # If whole batch is filtered, run in eval mode to avoid BN artifacts
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    out_eval = self.model(x)
                self.model.train(was_training)
                return _get_logits(out_eval)

        sample_w = reliable_mask.unsqueeze(1)  # (B, 1)

        # --- 4. Entropy over target prototypes for each branch ---
        local_loss_per_sample = None
        global_loss_per_sample = None
        if local_scores is not None and local_proto_identities is not None:
            local_loss_per_sample = self._branch_entropy_loss(
                sim_scores=local_scores,
                pred_class=pred_class,
                proto_identities=local_proto_identities,
                classifier=self.model.last_layer,
                sample_w=sample_w,
            )
        if global_scores is not None and global_proto_identities is not None:
            global_loss_per_sample = self._branch_entropy_loss(
                sim_scores=global_scores,
                pred_class=pred_class,
                proto_identities=global_proto_identities,
                classifier=self.model.last_layer_global,
                sample_w=sample_w,
            )
        if local_loss_per_sample is None:
            loss_per_sample = global_loss_per_sample
        elif global_loss_per_sample is None:
            loss_per_sample = local_loss_per_sample
        else:
            global_coe = float(getattr(self.model, 'global_coe', 0.5))
            loss_per_sample = ((1.0 - global_coe) * local_loss_per_sample +
                               global_coe * global_loss_per_sample)

        # --- 5. Confidence weighting ---
        with torch.no_grad():
            probs      = logits.softmax(dim=1)
            confidence = probs.max(dim=1)[0]
        proto_sample_loss = loss_per_sample * confidence * reliable_mask
        proto_loss = proto_sample_loss.sum() / \
                     (reliable_mask.sum() + 1e-8)

        entropy_per_sample = softmax_entropy(logits)
        logit_sample_weight = reliable_mask
        if self.shared_confidence_weighting:
            logit_sample_weight = logit_sample_weight * confidence
        output_sample_loss = entropy_per_sample * logit_sample_weight
        logit_loss = output_sample_loss.sum() / \
                     (reliable_mask.sum() + 1e-8)

        proto_reliability = self._prototype_reliability(
            local_scores, global_scores, pred_class,
            local_proto_identities, global_proto_identities,
        )
        output_probs = logits.softmax(dim=1)
        output_reliability = 1.0 - (
            -(output_probs * torch.log(output_probs.clamp_min(1e-8))).sum(dim=1)
            / math.log(output_probs.shape[1])
        )

        proto_weight = self.proto_weight
        logit_weight = self.logit_weight
        adaptive_lambda_per_sample = None
        proto_gradient_consistency = None
        output_gradient_consistency = None
        router_gate_fraction = None
        if self.adaptive_lambda:
            valid = reliable_mask.bool()
            if self.adaptive_lambda_strategy == 'mean_act_rel':
                # Simplest label-free combination: sample-wise average of the
                # two estimates, with no gradient routing.
                native_score = self._prototype_activation_margin(
                    local_scores, global_scores, pred_class,
                    local_proto_identities, global_proto_identities,
                )
                relative_score, proto_reliability, output_reliability = \
                    relative_evidence_lambda(
                        logits=logits,
                        branch_scores=(local_scores, global_scores),
                        branch_proto_identities=(
                            local_proto_identities, global_proto_identities
                        ),
                        topk=self.adaptive_topk,
                    )
                adaptive_score = 0.5 * (native_score + relative_score)
            elif self.adaptive_lambda_strategy == 'activation_margin':
                adaptive_score = self._prototype_activation_margin(
                    local_scores, global_scores, pred_class,
                    local_proto_identities, global_proto_identities,
                )
            elif self.adaptive_lambda_strategy == 'relative_evidence':
                adaptive_score, proto_reliability, output_reliability = \
                    relative_evidence_lambda(
                        logits=logits,
                        branch_scores=(local_scores, global_scores),
                        branch_proto_identities=(
                            local_proto_identities, global_proto_identities
                        ),
                        topk=self.adaptive_topk,
                    )
            elif self.adaptive_lambda_strategy in (
                    'source_free_router_absolute',
                    'source_free_router_coverage_absolute'):
                native_score = self._prototype_activation_margin(
                    local_scores, global_scores, pred_class,
                    local_proto_identities, global_proto_identities,
                )
                relative_score, relative_proto_rel, relative_output_rel = \
                    relative_evidence_lambda(
                        logits=logits,
                        branch_scores=(local_scores, global_scores),
                        branch_proto_identities=(
                            local_proto_identities, global_proto_identities
                        ),
                        topk=self.adaptive_topk,
                    )
                proto_gradient_consistency = self._gradient_consistency(
                    proto_sample_loss, reliable_mask
                )
                output_gradient_consistency = self._gradient_consistency(
                    output_sample_loss, reliable_mask
                )
                gradient_gate = (
                    proto_gradient_consistency
                    >= 2.0 * output_gradient_consistency.clamp_min(1e-8)
                )
                gradient_gate = gradient_gate & (
                    proto_gradient_consistency
                    >= self.router_min_consistency
                )
                if self.adaptive_lambda_strategy == \
                        'source_free_router_coverage_absolute':
                    # Sparse acceptance by the independent geometric filter is
                    # evidence against trusting a saturated output margin.
                    reliable_fraction = reliable_mask.float().mean()
                    coverage_guard = (
                        relative_output_rel[valid].mean() >= 0.90
                    ) & (reliable_fraction <= 0.40)
                    native_gate = torch.full_like(
                        native_score, dtype=torch.bool,
                        fill_value=(
                            bool(gradient_gate.item())
                            or bool(coverage_guard.item())
                        ),
                    )
                else:
                    native_gate = torch.full_like(
                        native_score, dtype=torch.bool,
                        fill_value=bool(gradient_gate.item()),
                    )
                adaptive_score = torch.where(
                    native_gate, native_score, relative_score
                )
                proto_reliability = relative_proto_rel
                output_reliability = relative_output_rel
                router_gate_fraction = native_gate[valid].float().mean().item()
            else:
                raise ValueError(f"Unknown adaptive lambda strategy: {self.adaptive_lambda_strategy}")
            adaptive_lambda_per_sample = adaptive_score.detach().clamp(
                self.lambda_min, self.lambda_max
            )
            current_lambda = adaptive_lambda_per_sample[valid].mean().item()
            # Sample-level coefficients enter the loss below; the reported
            # coefficient is the reliable-sample mean for diagnostics.
            proto_weight = current_lambda
            logit_weight = 1.0 - proto_weight

        need_grad_norms = self.gradient_normalize or self.record_diagnostics
        proto_grad_norm = self._gradient_norm(proto_loss) if need_grad_norms else None
        output_grad_norm = self._gradient_norm(logit_loss) if need_grad_norms else None
        proto_term = proto_loss
        output_term = logit_loss
        if self.gradient_normalize:
            proto_term = proto_loss / (proto_grad_norm.detach() + 1e-8)
            output_term = logit_loss / (output_grad_norm.detach() + 1e-8)

        if self.adaptive_lambda:
            proto_sample_term = proto_sample_loss
            output_sample_term = output_sample_loss
            if self.gradient_normalize:
                proto_sample_term = proto_sample_term / (proto_grad_norm.detach() + 1e-8)
                output_sample_term = output_sample_term / (output_grad_norm.detach() + 1e-8)
            total_loss = (
                adaptive_lambda_per_sample * proto_sample_term
                + (1.0 - adaptive_lambda_per_sample) * output_sample_term
            ).sum() / (reliable_mask.sum() + 1e-8)
        else:
            total_loss = proto_weight * proto_term + logit_weight * output_term

        if self.record_diagnostics or self.adaptive_lambda:
            valid = reliable_mask.bool()
            self.adaptation_stats['proto_loss'].append(float(proto_loss.detach().item()))
            self.adaptation_stats['output_loss'].append(float(logit_loss.detach().item()))
            if proto_grad_norm is not None:
                self.adaptation_stats['proto_grad_norm'].append(float(proto_grad_norm.item()))
                self.adaptation_stats['output_grad_norm'].append(float(output_grad_norm.item()))
            self.adaptation_stats['adaptive_lambda'].append(float(proto_weight))
            if self.adaptive_lambda:
                self.adaptation_stats['adaptive_lambda_raw'].append(float(current_lambda))
            self.adaptation_stats['proto_signal_reliability'].append(
                float(proto_reliability[valid].mean().detach().item()))
            self.adaptation_stats['output_signal_reliability'].append(
                float(output_reliability[valid].mean().detach().item()))
            if proto_gradient_consistency is not None:
                self.adaptation_stats['proto_gradient_consistency'].append(
                    float(proto_gradient_consistency.item())
                )
                self.adaptation_stats['output_gradient_consistency'].append(
                    float(output_gradient_consistency.item())
                )
                self.adaptation_stats['adaptive_router_gate'].append(
                    float(router_gate_fraction)
                )

        # --- 6. Backward ---
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.adaptation_stats['total_updates'] += 1

        return logits

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_proto_activations(self, out):
        """Extract pooled local/global prototype activations from ProtoPFormer."""
        if not isinstance(out, tuple) or len(out) < 2:
            return None, None

        aux = out[1]
        if not isinstance(aux, (tuple, list)):
            return None, None

        local_acts = None
        global_acts = None

        # ProtoPFormer exposes different auxiliary tuples by mode:
        #   eval:  (..., logits_local, local_acts, global_acts)  [len=6]
        #   train: (..., original_fea_len, local_acts, global_acts) [len=7]
        if len(aux) >= 7 and isinstance(aux[5], torch.Tensor):
            local_acts = aux[5]
        elif len(aux) == 6 and isinstance(aux[4], torch.Tensor):
            local_acts = aux[4]
        elif len(aux) >= 3 and isinstance(aux[2], torch.Tensor) and aux[2].dim() == 4:
            local_acts = self._consensus(aux[2].flatten(2))

        if len(aux) >= 7 and isinstance(aux[6], torch.Tensor):
            global_acts = aux[6]
        elif len(aux) == 6 and isinstance(aux[5], torch.Tensor):
            global_acts = aux[5]

        return local_acts, global_acts

    def _normalize_similarity(self, raw_scores):
        if raw_scores is None:
            return None
        raw_scores = F.relu(raw_scores)
        return torch.sigmoid((raw_scores - self.sigmoid_center) / max(self.sigmoid_temp, 1e-6))

    def _branch_entropy_loss(self, sim_scores, pred_class, proto_identities,
                             classifier, sample_w):
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        eps = 1e-6
        proto_probs = torch.clamp(sim_scores * target_mask, eps, 1.0 - eps)
        entropy = -(proto_probs * torch.log(proto_probs) +
                    (1.0 - proto_probs) * torch.log(1.0 - proto_probs))

        if self.use_importance:
            class_w = classifier.weight[pred_class]
            imp = torch.abs(class_w) * target_mask
            imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-8)
            weighted_entropy = entropy * target_mask * imp * sample_w
            return weighted_entropy.sum(dim=1)

        masked_e = entropy * target_mask * sample_w
        return masked_e.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)

    def _branch_reliability(self, sim_scores, pred_class, proto_identities):
        """Label-free concentration and strength of predicted-class prototypes."""
        if sim_scores is None or proto_identities is None:
            return None
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        target_scores = sim_scores * target_mask
        q = target_scores / target_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=1)
        num_target = target_mask.sum(dim=1).clamp_min(2.0)
        concentration = 1.0 - entropy / torch.log(num_target)
        strength = target_scores.max(dim=1).values / sim_scores.max(dim=1).values.clamp_min(1e-8)
        return (concentration * strength).clamp(0.0, 1.0)

    def _prototype_reliability(self, local_scores, global_scores, pred_class,
                               local_proto_identities, global_proto_identities):
        local = self._branch_reliability(local_scores, pred_class, local_proto_identities)
        global_ = self._branch_reliability(global_scores, pred_class, global_proto_identities)
        if local is None:
            return global_
        if global_ is None:
            return local
        global_coe = float(getattr(self.model, 'global_coe', 0.5))
        return (1.0 - global_coe) * local + global_coe * global_

    def _branch_activation_margin(self, sim_scores, pred_class, proto_identities):
        """Normalized top-target distance from the binary-entropy boundary 0.5."""
        if sim_scores is None or proto_identities is None:
            return None
        target_mask = proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)
        masked = sim_scores.masked_fill(~target_mask, float('-inf'))
        k = min(self.adaptive_topk, int(target_mask.sum(dim=1).min().item()))
        top_target = masked.topk(max(k, 1), dim=1).values
        delta = (top_target - 0.5).abs().mean(dim=1)
        return (delta / max(self.adaptive_delta0, 1e-8)).clamp(0.0, 1.0)

    def _prototype_activation_margin(self, local_scores, global_scores, pred_class,
                                     local_proto_identities, global_proto_identities):
        local = self._branch_activation_margin(local_scores, pred_class, local_proto_identities)
        global_ = self._branch_activation_margin(global_scores, pred_class, global_proto_identities)
        if local is None:
            return global_
        if global_ is None:
            return local
        global_coe = float(getattr(self.model, 'global_coe', 0.5))
        return (1.0 - global_coe) * local + global_coe * global_

    def _gradient_norm(self, loss):
        params = [
            p for group in self.optimizer.param_groups for p in group['params']
            if p.requires_grad
        ]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        squared = [g.detach().float().pow(2).sum() for g in grads if g is not None]
        if not squared:
            return loss.detach().new_tensor(0.0)
        return torch.stack(squared).sum().sqrt()

    def _gradient_consistency(self, per_sample_loss, reliable_mask):
        """Scale-invariant gradient agreement across two reliable halves."""
        indices = reliable_mask.bool().nonzero(as_tuple=False).flatten()
        if indices.numel() < 4:
            return per_sample_loss.detach().new_tensor(0.0)
        first = indices[::2]
        second = indices[1::2]
        first_loss = per_sample_loss[first].mean()
        second_loss = per_sample_loss[second].mean()
        params = self._optimizer_params()
        first_grads = torch.autograd.grad(
            first_loss, params, retain_graph=True, allow_unused=True
        )
        second_grads = torch.autograd.grad(
            second_loss, params, retain_graph=True, allow_unused=True
        )
        dot = first_loss.detach().new_tensor(0.0, dtype=torch.float32)
        first_sq = dot.clone()
        second_sq = dot.clone()
        for first_grad, second_grad in zip(first_grads, second_grads):
            if first_grad is None or second_grad is None:
                continue
            first_float = first_grad.detach().float()
            second_float = second_grad.detach().float()
            dot = dot + (first_float * second_float).sum()
            first_sq = first_sq + first_float.pow(2).sum()
            second_sq = second_sq + second_float.pow(2).sum()
        first_norm = first_sq.sqrt()
        second_norm = second_sq.sqrt()
        cosine = dot / (first_norm * second_norm + 1e-12)
        norm_balance = (
            2.0 * torch.minimum(first_norm, second_norm)
            / (first_norm + second_norm + 1e-12)
        )
        return cosine.clamp(0.0, 1.0) * norm_balance.clamp(0.0, 1.0)

    def _optimizer_params(self):
        """Return each optimizer parameter once, preserving group order."""
        params = []
        seen = set()
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad and id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        return params

    def _consensus(self, similarities):
        """Aggregate over the spatial patches (dim=2) by the maximum."""
        if similarities.dim() < 3:
            return similarities

        # similarities: [B, P, K]
        return similarities.max(dim=2)[0]

    def forward_no_adapt(self, x):
        return _forward_eval(self.model, x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_proto_tta(model, lr=1e-3, steps=1,
                    use_importance=True,
                    geo_filter_threshold=0.3,
                    adaptation_mode='layernorm_attn_bias',
                    prototype_branch='both',
                    similarity_mapping='sigmoid',
                    sigmoid_center=2.0,
                    sigmoid_temp=1.0,
                    proto_weight=1.0,
                    logit_weight=0.0,
                    shared_confidence_weighting=False,
                    gradient_normalize=False,
                    adaptive_lambda=False,
                    adaptive_lambda_strategy=None,
                    adaptive_delta0=0.25,
                    adaptive_topk=3,
                    router_min_consistency=0.25,
                    lambda_min=0.05,
                    lambda_max=0.95,
                    record_diagnostics=False,
                    record_vlm_evidence=False,
                    model_mode='train'):
    """Factory: configure + wrap model with ProtoTTA."""
    model = configure_model(model, adaptation_mode, model_mode=model_mode)
    params, _ = collect_params(model, adaptation_mode)
    if not params:
        # Fallback to layernorm_only
        model = configure_model(model, 'layernorm_only', model_mode=model_mode)
        params, _ = collect_params(model, 'layernorm_only')
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return ProtoTTA(
        model, optimizer, steps=steps,
        use_importance=use_importance,
        geo_filter_threshold=geo_filter_threshold,
        prototype_branch=prototype_branch,
        similarity_mapping=similarity_mapping,
        sigmoid_center=sigmoid_center,
        sigmoid_temp=sigmoid_temp,
        proto_weight=proto_weight,
        logit_weight=logit_weight,
        shared_confidence_weighting=shared_confidence_weighting,
        gradient_normalize=gradient_normalize,
        adaptive_lambda=adaptive_lambda,
        adaptive_lambda_strategy=adaptive_lambda_strategy,
        adaptive_delta0=adaptive_delta0,
        adaptive_topk=adaptive_topk,
        router_min_consistency=router_min_consistency,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        record_diagnostics=record_diagnostics,
        record_vlm_evidence=record_vlm_evidence,
    )
