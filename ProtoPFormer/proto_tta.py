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

def configure_model(model, adaptation_mode='layernorm_only', model_mode='train'):
    """Configure adaptation parameters and the explicitly requested model mode.

    ``train`` matches the upstream Tent/EATA implementations and the historical
    paper runs. ``eval`` is a deterministic transformer ablation that disables
    DropPath/dropout; it must not be silently reported as standard Tent.
    """
    if model_mode == 'train':
        model.train()
    elif model_mode == 'eval':
        model.eval()
    else:
        raise ValueError(f"Unknown adaptation model mode: {model_mode}")
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
                 shared_confidence_weighting=False,
                 gradient_normalize=False,
                 adaptive_lambda=False,
                 adaptive_lambda_strategy='relative_reliability',
                 adaptive_delta0=0.25,
                 adaptive_topk=3,
                 lambda_ema_momentum=0.9,
                 lambda_min=0.05,
                 lambda_max=0.95,
                 record_diagnostics=False,
                 lambda_search=False,
                 lambda_search_radius=0.1,
                 lambda_search_teacher_temp=0.5,
                 lambda_search_min_improvement=0.0,
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
        self.shared_confidence_weighting = shared_confidence_weighting
        self.gradient_normalize = gradient_normalize
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_lambda_strategy = adaptive_lambda_strategy
        self.adaptive_delta0 = adaptive_delta0
        self.adaptive_topk = adaptive_topk
        self.lambda_ema_momentum = lambda_ema_momentum
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.record_diagnostics = record_diagnostics
        self.lambda_search = lambda_search
        self.lambda_search_radius = lambda_search_radius
        self.lambda_search_teacher_temp = lambda_search_teacher_temp
        self.lambda_search_min_improvement = lambda_search_min_improvement
        self.lambda_ema = None

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
            'proto_loss': [],
            'output_loss': [],
            'proto_grad_norm': [],
            'output_grad_norm': [],
            'adaptive_lambda': [],
            'adaptive_lambda_raw': [],
            'proto_signal_reliability': [],
            'output_signal_reliability': [],
            'lambda_search_selected': [],
            'lambda_search_base_score': [],
            'lambda_search_selected_score': [],
            'lambda_search_accepted': 0,
            'lambda_search_rejected': 0,
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
                    was_training = self.model.training
                    self.model.eval()
                    with torch.no_grad():
                        out_eval = self.model(x)
                    self.model.train(was_training)
                    return _get_logits(out_eval)
        else:
            reliable_mask = torch.ones(x.size(0), device=x.device)
            adapted = x.size(0)
            self.adaptation_stats['adapted_samples'] += adapted

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

        entropy_per_sample = softmax_entropy(logits)
        logit_sample_weight = reliable_mask
        if self.shared_confidence_weighting and self.use_confidence:
            logit_sample_weight = logit_sample_weight * confidence
        logit_loss = (entropy_per_sample * logit_sample_weight).sum() / \
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
        if self.adaptive_lambda:
            valid = reliable_mask.bool()
            if self.adaptive_lambda_strategy == 'activation_margin':
                adaptive_score = self._prototype_activation_margin(
                    local_scores, global_scores, pred_class,
                    local_proto_identities, global_proto_identities,
                )
            elif self.adaptive_lambda_strategy == 'relative_reliability':
                adaptive_score = proto_reliability / (
                    proto_reliability + output_reliability + 1e-8
                )
            else:
                raise ValueError(f"Unknown adaptive lambda strategy: {self.adaptive_lambda_strategy}")
            current_lambda = adaptive_score[valid].mean().detach().item()
            if self.lambda_ema is None:
                self.lambda_ema = current_lambda
            else:
                self.lambda_ema = (self.lambda_ema_momentum * self.lambda_ema +
                                   (1.0 - self.lambda_ema_momentum) * current_lambda)
            proto_weight = min(self.lambda_max, max(self.lambda_min, self.lambda_ema))
            logit_weight = 1.0 - proto_weight

        need_grad_norms = self.gradient_normalize or self.record_diagnostics
        proto_grad_norm = self._gradient_norm(proto_loss) if need_grad_norms else None
        output_grad_norm = self._gradient_norm(logit_loss) if need_grad_norms else None
        proto_term = proto_loss
        output_term = logit_loss
        if self.gradient_normalize:
            proto_term = proto_loss / (proto_grad_norm.detach() + 1e-8)
            output_term = logit_loss / (output_grad_norm.detach() + 1e-8)

        search_result = None
        if self.lambda_search:
            if not self.adaptive_lambda:
                raise ValueError('lambda_search requires adaptive_lambda=True')
            search_result = self._select_and_apply_lambda_update(
                x=x,
                base_lambda=proto_weight,
                proto_term=proto_term,
                output_term=output_term,
            )
            if search_result['accepted']:
                proto_weight = search_result['selected_lambda']
                logit_weight = 1.0 - proto_weight
                self.adaptation_stats['total_updates'] += 1
                self.adaptation_stats['lambda_search_accepted'] += 1
            else:
                # This batch contributed no parameter update.
                self.adaptation_stats['adapted_samples'] -= adapted
                self.adaptation_stats['lambda_search_rejected'] += 1

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
            if search_result is not None:
                selected = search_result['selected_lambda']
                self.adaptation_stats['lambda_search_selected'].append(
                    None if selected is None else float(selected)
                )
                self.adaptation_stats['lambda_search_base_score'].append(
                    float(search_result['base_score'])
                )
                self.adaptation_stats['lambda_search_selected_score'].append(
                    float(search_result['selected_score'])
                )

        # --- 6. Backward ---
        if not self.lambda_search:
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

    @staticmethod
    def _set_weighted_grads(params, proto_grads, output_grads, proto_weight):
        for param, proto_grad, output_grad in zip(params, proto_grads, output_grads):
            if proto_grad is None and output_grad is None:
                param.grad = None
                continue
            grad = torch.zeros_like(param)
            if proto_grad is not None:
                grad.add_(proto_grad, alpha=proto_weight)
            if output_grad is not None:
                grad.add_(output_grad, alpha=1.0 - proto_weight)
            param.grad = grad

    def _restore_search_state(self, params, param_state, optimizer_state):
        with torch.no_grad():
            for param, value in zip(params, param_state):
                param.copy_(value)
        self.optimizer.load_state_dict(deepcopy(optimizer_state))
        self.optimizer.zero_grad()

    def _search_score(self, teacher_probs, candidate_logits):
        """Cross-view soft pseudo-label score; lower is better."""
        candidate_log_probs = candidate_logits.log_softmax(dim=1)
        return -(teacher_probs * candidate_log_probs).sum(dim=1).mean()

    def _select_and_apply_lambda_update(self, x, base_lambda, proto_term, output_term):
        """Select a local lambda by virtual Adam updates and commit it safely.

        The candidates are ``lambda_hat +/- radius`` and ``lambda_hat``.  Each
        virtual update starts from the exact same model and optimizer state.
        It is scored on a deterministic horizontal-flip view against a frozen,
        sharpened pre-update teacher.  The unchanged model is an explicit
        candidate, so an update is committed only when it improves this
        label-free cross-view score.
        """
        params = self._optimizer_params()
        proto_grads = torch.autograd.grad(
            proto_term, params, retain_graph=True, allow_unused=True
        )
        output_grads = torch.autograd.grad(
            output_term, params, retain_graph=False, allow_unused=True
        )
        proto_grads = tuple(None if grad is None else grad.detach() for grad in proto_grads)
        output_grads = tuple(None if grad is None else grad.detach() for grad in output_grads)

        param_state = [param.detach().clone() for param in params]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        was_training = self.model.training
        search_view = torch.flip(x, dims=(-1,))

        self.model.eval()
        with torch.no_grad():
            teacher_logits = _get_logits(self.model(x))
            teacher_probs = (
                teacher_logits / max(self.lambda_search_teacher_temp, 1e-6)
            ).softmax(dim=1)
            base_logits = _get_logits(self.model(search_view))
            base_score = float(self._search_score(teacher_probs, base_logits).item())

        candidates = sorted({
            min(self.lambda_max, max(self.lambda_min, base_lambda - self.lambda_search_radius)),
            min(self.lambda_max, max(self.lambda_min, base_lambda)),
            min(self.lambda_max, max(self.lambda_min, base_lambda + self.lambda_search_radius)),
        })
        best_lambda = None
        best_score = base_score

        for candidate in candidates:
            self._restore_search_state(params, param_state, optimizer_state)
            self._set_weighted_grads(params, proto_grads, output_grads, candidate)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.eval()
            with torch.no_grad():
                candidate_logits = _get_logits(self.model(search_view))
                score = float(self._search_score(teacher_probs, candidate_logits).item())
            if score < best_score - self.lambda_search_min_improvement:
                best_score = score
                best_lambda = float(candidate)

        self._restore_search_state(params, param_state, optimizer_state)
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        accepted = best_lambda is not None
        if accepted:
            self._set_weighted_grads(params, proto_grads, output_grads, best_lambda)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            'accepted': accepted,
            'selected_lambda': best_lambda,
            'base_score': base_score,
            'selected_score': best_score,
        }

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
                    logit_weight=0.0,
                    shared_confidence_weighting=False,
                    gradient_normalize=False,
                    adaptive_lambda=False,
                    adaptive_lambda_strategy='relative_reliability',
                    adaptive_delta0=0.25,
                    adaptive_topk=3,
                    lambda_ema_momentum=0.9,
                    lambda_min=0.05,
                    lambda_max=0.95,
                    record_diagnostics=False,
                    lambda_search=False,
                    lambda_search_radius=0.1,
                    lambda_search_teacher_temp=0.5,
                    lambda_search_min_improvement=0.0,
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
        shared_confidence_weighting=shared_confidence_weighting,
        gradient_normalize=gradient_normalize,
        adaptive_lambda=adaptive_lambda,
        adaptive_lambda_strategy=adaptive_lambda_strategy,
        adaptive_delta0=adaptive_delta0,
        adaptive_topk=adaptive_topk,
        lambda_ema_momentum=lambda_ema_momentum,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        record_diagnostics=record_diagnostics,
        lambda_search=lambda_search,
        lambda_search_radius=lambda_search_radius,
        lambda_search_teacher_temp=lambda_search_teacher_temp,
        lambda_search_min_improvement=lambda_search_min_improvement,
    )
