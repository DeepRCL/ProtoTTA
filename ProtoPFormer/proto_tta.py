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
  - LayerNorm layers inside the ViT backbone for TTA

Only `proto_imp_conf_v3` (best variant) + base `tent` + `eata` are implemented.
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
    """Collect parameters to adapt (LayerNorms, Biases, or Prototypes)."""
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

    if 'proto' in adaptation_mode:
        for attr in ['prototype_vectors', 'prototype_vectors_global']:
            if hasattr(model, attr):
                p = getattr(model, attr)
                params.append(p)
                names.append(attr)

    return params, names

def configure_model(model, adaptation_mode='layernorm_only'):
    """Put model in train mode and enable only the target parameters."""
    model.train()
    model.requires_grad_(False)

    params, names = collect_params(model, adaptation_mode)
    for p in params:
        p.requires_grad = True

    # Ensure BN layers use current batch stats (standard TTA practice)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
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


def setup_tent(model, lr=1e-3, steps=1, episodic=False):
    model = configure_model(model, 'layernorm_only')
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


def setup_eata(model, fishers=None, lr=1e-3, steps=1, episodic=False):
    model = configure_model(model, 'layernorm_only')
    params, _ = collect_params(model, 'layernorm_only')
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return EATA(model, optimizer, fishers=fishers, steps=steps, episodic=episodic)


# ============================================================================
# ProtoTTA  (proto_imp_conf_v3 — best variant)
# ============================================================================
# ProtoPFormer does NOT expose sub-prototype similarities like ProtoViT.
# Its prototype activations come from distance_2_similarity(distances), where
# `distances` is shape (B, num_prototypes, H, W).
# We use the global-max-pooled activations as "similarities" ∈ (0, 1).
#
# proto_imp_conf_v3 config (from evaluate_robustness.py):
#   use_importance=True, use_confidence=True
#   use_geometric_filter=True, geo_filter_threshold=0.92
#   consensus_strategy='top_k_mean', consensus_ratio=0.5
#   adaptation_mode='layernorm_attn_bias'
#   use_ensemble_entropy=False  ← v3 distinction
#   reset_mode=None (no reset)
# ============================================================================

class ProtoTTA(nn.Module):
    """ProtoTTA for ProtoPFormer — prototype-similarity entropy minimisation.

    Implements the full proto_imp_conf_v3 variant:
      - Importance weighting (last_layer weights)
      - Confidence weighting (softmax confidence)
      - Geometric filtering (filter samples far from all prototypes)
      - Consensus aggregation (top-k mean over spatial prototype activations)
      - No ensemble entropy (v3)
    """

    def __init__(self, model, optimizer,
                 steps=1, episodic=False,
                 # Importance & confidence
                 use_importance=True,
                 use_confidence=True,
                 adapt_all_prototypes=False,
                 # Geometric filter
                 use_geometric_filter=True,
                 geo_filter_threshold=0.3,
                 # Consensus
                 consensus_strategy='max',
                 consensus_ratio=0.5,
                 # Reset
                 reset_mode=None,
                 reset_frequency=10,
                 confidence_threshold=0.7,
                 ema_alpha=0.999,
                 use_branch_agreement=False,
                 prototype_branch='both',
                 similarity_mapping='sigmoid',
                 sigmoid_center=2.0,
                 sigmoid_temp=1.0,
                 proto_weight=1.0,
                 logit_weight=0.0,
                 ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.use_importance = use_importance
        self.use_confidence = use_confidence
        self.adapt_all_prototypes = adapt_all_prototypes
        self.use_geometric_filter = use_geometric_filter
        self.geo_filter_threshold = geo_filter_threshold
        self.consensus_strategy = consensus_strategy
        self.consensus_ratio = consensus_ratio

        self.reset_mode = 'episodic' if episodic else (reset_mode or 'none')
        self.reset_frequency = reset_frequency
        self.confidence_threshold = confidence_threshold
        self.ema_alpha = ema_alpha
        self.use_branch_agreement = use_branch_agreement
        self.prototype_branch = prototype_branch
        self.similarity_mapping = similarity_mapping
        self.sigmoid_center = sigmoid_center
        self.sigmoid_temp = sigmoid_temp
        self.proto_weight = proto_weight
        self.logit_weight = logit_weight

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(model, optimizer)

        epsilon = float(getattr(model, 'epsilon', 1e-4))
        self.similarity_scale = max(-math.log(epsilon), 1.0)

        self.batch_count = 0
        self.confidence_history = []
        self.ema_state = None

        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'total_updates': 0,
            'branch_agreement_samples': 0,
            'avg_reliability': [],
        }

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x):
        if self._should_reset(x):
            self.reset()

        self.adaptation_stats['total_samples'] += x.size(0)

        for _ in range(self.steps):
            logits = self._forward_and_adapt(x)

        self._update_tracking(logits)
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

        if self.prototype_branch == 'local':
            global_raw = None
            global_scores = None
        elif self.prototype_branch == 'global':
            local_raw = None
            local_scores = None

        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            local_proto_identities = None
            if local_scores is not None:
                local_pci = self.model.prototype_class_identity.to(logits.device)
                local_proto_identities = local_pci.argmax(dim=1)

            branch_agreement = torch.ones_like(pred_class, dtype=torch.bool)
            local_branch_pred = None
            if local_raw is not None:
                local_branch_pred = self.model.last_layer(local_raw).argmax(dim=1)
            global_proto_identities = None

            if global_scores is not None and hasattr(self.model, 'prototype_class_identity_global'):
                global_pci = self.model.prototype_class_identity_global.to(logits.device)
                global_proto_identities = global_pci.argmax(dim=1)
                global_branch_pred = self.model.last_layer_global(global_raw).argmax(dim=1)
                if self.use_branch_agreement and local_branch_pred is not None:
                    branch_agreement = (local_branch_pred == pred_class) & (global_branch_pred == pred_class)
                self.adaptation_stats['branch_agreement_samples'] += int(branch_agreement.sum().item())

        # --- 3. Geometric filter ---
        if self.use_geometric_filter:
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
                if self.use_branch_agreement:
                    reliable_mask = reliable_mask & branch_agreement
                reliable_mask = reliable_mask.float()
                
                # Debug logging (every 50 batches)
                if self.batch_count % 50 == 0 and self.batch_count > 0:
                     print(f" [Batch {self.batch_count + 1}] Max Sim Avg: {reliability_score.mean():.3f}, Threshold: {self.geo_filter_threshold}")

                adapted = int(reliable_mask.sum().item())
                self.adaptation_stats['adapted_samples'] += adapted
                self.adaptation_stats['avg_reliability'].append(float(reliability_score.mean().item()))
                if reliable_mask.sum() == 0:
                    # If whole batch is filtered, run in eval mode to avoid BN artifacts
                    self.model.eval()
                    with torch.no_grad():
                        out_eval = self.model(x)
                    self.model.train()
                    return _get_logits(out_eval)
        else:
            reliable_mask = torch.ones(x.size(0), device=x.device)
            self.adaptation_stats['adapted_samples'] += x.size(0)

        self.adaptation_stats['total_updates'] += 1
        sample_w = reliable_mask.unsqueeze(1)  # (B, 1)

        # --- 4. Entropy over target prototypes for each branch ---
        loss_per_sample = None
        if local_scores is not None and local_proto_identities is not None:
            loss_per_sample = self._branch_entropy_loss(
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
            if loss_per_sample is None:
                loss_per_sample = global_loss_per_sample
            else:
                global_coe = float(getattr(self.model, 'global_coe', 0.5))
                loss_per_sample = ((1.0 - global_coe) * loss_per_sample +
                                   global_coe * global_loss_per_sample)

        # --- 5. Confidence weighting ---
        if self.use_confidence:
            with torch.no_grad():
                probs      = logits.softmax(dim=1)
                confidence = probs.max(dim=1)[0]
            proto_loss = (loss_per_sample * confidence * reliable_mask).sum() / \
                         (reliable_mask.sum() + 1e-8)
        else:
            proto_loss = (loss_per_sample * reliable_mask).sum() / \
                         (reliable_mask.sum() + 1e-8)

        logit_loss = torch.tensor(0.0, device=logits.device)
        if self.logit_weight > 0:
            entropy_per_sample = softmax_entropy(logits)
            logit_loss = (entropy_per_sample * reliable_mask).sum() / \
                         (reliable_mask.sum() + 1e-8)

        total_loss = self.proto_weight * proto_loss + self.logit_weight * logit_loss

        # --- 6. Backward ---
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

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

        if len(aux) >= 6 and isinstance(aux[5], torch.Tensor):
            local_acts = aux[5]
        elif len(aux) >= 3 and isinstance(aux[2], torch.Tensor) and aux[2].dim() == 4:
            local_acts = self._consensus(aux[2].flatten(2))

        if len(aux) >= 7 and isinstance(aux[6], torch.Tensor):
            global_acts = aux[6]

        return local_acts, global_acts

    def _normalize_similarity(self, raw_scores):
        if raw_scores is None:
            return None
        raw_scores = F.relu(raw_scores)
        if self.similarity_mapping == 'linear':
            return torch.clamp(raw_scores / self.similarity_scale, 0.0, 1.0)
        if self.similarity_mapping == 'sigmoid':
            return torch.sigmoid((raw_scores - self.sigmoid_center) / max(self.sigmoid_temp, 1e-6))
        raise ValueError(f"Unknown similarity_mapping: {self.similarity_mapping}")

    def _branch_entropy_loss(self, sim_scores, pred_class, proto_identities, classifier, sample_w):
        if self.adapt_all_prototypes:
            target_mask = torch.ones_like(sim_scores)
        else:
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

    def _consensus(self, similarities):
        """Aggregate over the spatial patches (dim=2)."""
        if similarities.dim() < 3:
            return similarities

        # similarities: [B, P, K]
        if self.consensus_strategy == 'max':
            agg = similarities.max(dim=2)[0]
        elif self.consensus_strategy == 'top_k_mean':
            K = similarities.shape[2]
            top_k = max(1, int(K * self.consensus_ratio))
            top_sims = torch.topk(similarities, k=top_k, dim=2)[0]
            agg = top_sims.mean(dim=2)
        else:
            agg = similarities.mean(dim=2)
        return agg

    def _should_reset(self, x):
        if self.reset_mode == 'episodic':
            return True
        if self.reset_mode == 'none':
            return False
        if self.reset_mode == 'periodic':
            return self.batch_count > 0 and self.batch_count % self.reset_frequency == 0
        if self.reset_mode == 'confidence':
            if len(self.confidence_history) >= 5:
                return (sum(self.confidence_history[-5:]) / 5) < self.confidence_threshold
            return False
        return False

    def _update_tracking(self, logits):
        self.batch_count += 1
        with torch.no_grad():
            probs = logits.softmax(dim=1)
            self.confidence_history.append(probs.max(dim=1)[0].mean().item())
            if len(self.confidence_history) > 50:
                self.confidence_history = self.confidence_history[-50:]

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def forward_no_adapt(self, x):
        return _forward_eval(self.model, x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_proto_tta(model, lr=1e-3, steps=1, episodic=False,
                    use_importance=True, use_confidence=True,
                    adapt_all_prototypes=False,
                    use_geometric_filter=True, geo_filter_threshold=0.3,
                    consensus_strategy='max', consensus_ratio=0.5,
                    adaptation_mode='layernorm_attn_bias',
                    reset_mode=None, reset_frequency=10,
                    confidence_threshold=0.7, ema_alpha=0.999,
                    use_branch_agreement=False,
                    prototype_branch='both',
                    similarity_mapping='sigmoid',
                    sigmoid_center=2.0,
                    sigmoid_temp=1.0,
                    proto_weight=1.0,
                    logit_weight=0.0):
    """Factory: configure + wrap model with ProtoTTA."""
    model = configure_model(model, adaptation_mode)
    params, _ = collect_params(model, adaptation_mode)
    if not params:
        # Fallback to layernorm_only
        model = configure_model(model, 'layernorm_only')
        params, _ = collect_params(model, 'layernorm_only')
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return ProtoTTA(
        model, optimizer, steps=steps, episodic=episodic,
        use_importance=use_importance,
        use_confidence=use_confidence,
        adapt_all_prototypes=adapt_all_prototypes,
        use_geometric_filter=use_geometric_filter,
        geo_filter_threshold=geo_filter_threshold,
        consensus_strategy=consensus_strategy,
        consensus_ratio=consensus_ratio,
        reset_mode=reset_mode,
        reset_frequency=reset_frequency,
        confidence_threshold=confidence_threshold,
        ema_alpha=ema_alpha,
        use_branch_agreement=use_branch_agreement,
        prototype_branch=prototype_branch,
        similarity_mapping=similarity_mapping,
        sigmoid_center=sigmoid_center,
        sigmoid_temp=sigmoid_temp,
        proto_weight=proto_weight,
        logit_weight=logit_weight,
    )
