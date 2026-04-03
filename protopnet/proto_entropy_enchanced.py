"""
Enhanced Prototype-aware Entropy Minimization (ProtoEntropy++) for ProtoPNet.

Improvements over base ProtoEntropy:
1. SAR-style entropy filtering (reliable sample selection)
2. Optional Sharpness-Aware Minimization (SAM) optimizer
3. Hybrid loss: binary proto entropy + softmax entropy blend
4. Adaptive geometric threshold based on batch statistics
5. EMA model updates for stability

Usage:
    # Create enhanced ProtoEntropy with SAM
    model = setup_proto_entropy_enhanced(model, use_sam=True)
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for reset capability."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict()) if optimizer else None
    return model_state, optimizer_state


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper.
    
    Seeks flatter minima that generalize better, especially useful for blur corruptions.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute epsilon (perturbation) and move to w + epsilon."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Move to w + epsilon
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights and apply gradient update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Restore original weights
        self.base_optimizer.step()  # Apply gradient at original position
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Standard step (for compatibility)."""
        assert closure is not None, "SAM requires closure for step()"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class ProtoEntropyEnhanced(nn.Module):
    """Enhanced Prototype-aware entropy minimization with SAR-style improvements.
    
    Key improvements:
    1. Entropy filtering (SAR-style) - only adapt on reliable samples
    2. SAM optimizer support - sharpness-aware for better blur robustness
    3. Hybrid loss - blend of binary proto entropy and softmax entropy
    4. Adaptive thresholds - adjust filtering based on batch statistics
    5. EMA updates - exponential moving average for stability
    """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 # Loss weights
                 alpha_proto=1.0, alpha_softmax=0.0, alpha_separation=0.0,
                 # Filtering
                 use_entropy_filter=True, entropy_margin_scale=0.4,
                 use_geometric_filter=False, geo_filter_threshold=0.3,
                 use_adaptive_threshold=False,
                 # Prototype settings
                 use_prototype_importance=False,
                 use_confidence_weighting=False,
                 adapt_all_prototypes=False,
                 # SAM settings
                 use_sam=False,
                 # EMA settings
                 use_ema=False, ema_alpha=0.999,
                 # Reset settings
                 reset_mode=None, reset_frequency=10,
                 # SAR-style recovery
                 use_model_recovery=False, recovery_threshold=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ProtoEntropyEnhanced requires >= 1 step(s)"
        self.episodic = episodic

        # Loss weights
        self.alpha_proto = alpha_proto  # Weight for binary prototype entropy
        self.alpha_softmax = alpha_softmax  # Weight for softmax entropy (SAR-style)
        self.alpha_separation = alpha_separation  # Weight for pushing non-target protos
        
        # Filtering settings
        self.use_entropy_filter = use_entropy_filter  # SAR-style entropy filtering
        self.entropy_margin_scale = entropy_margin_scale  # 0.4 * log(num_classes) typical
        self.use_geometric_filter = use_geometric_filter
        self.geo_filter_threshold = geo_filter_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        
        # Prototype settings
        self.use_prototype_importance = use_prototype_importance
        self.use_confidence_weighting = use_confidence_weighting
        self.adapt_all_prototypes = adapt_all_prototypes
        
        # SAM settings
        self.use_sam = use_sam
        
        # EMA settings
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.ema_state = None
        
        # Reset settings
        if reset_mode is None:
            self.reset_mode = 'episodic' if episodic else 'none'
        else:
            self.reset_mode = reset_mode
        self.reset_frequency = reset_frequency
        
        # SAR-style model recovery
        self.use_model_recovery = use_model_recovery
        self.recovery_threshold = recovery_threshold
        self.ema_loss = None  # Moving average of loss for recovery

        # Save model/optimizer state for reset
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Tracking
        self.batch_count = 0
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'filtered_by_entropy': 0,
            'filtered_by_geo': 0,
            'total_updates': 0,
            'model_resets': 0,
        }
        self.geo_filter_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'min_similarities': [],
            'max_similarities': [],
            'avg_similarities': []
        }

    def forward(self, x):
        if self.reset_mode == 'episodic':
            self.reset()

        batch_size = x.size(0)
        self.adaptation_stats['total_samples'] += batch_size

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        
        self.batch_count += 1
        
        # Handle periodic reset
        if self.reset_mode == 'periodic' and self.batch_count % self.reset_frequency == 0:
            self.reset()
        
        # EMA update
        if self.use_ema:
            self._apply_ema_update()
        
        return outputs

    def _get_model_components(self):
        """Get prototype_class_identity and last_layer from model."""
        if hasattr(self.model, 'core'):
            proto_class_identity = self.model.core.prototype_class_identity
            last_layer = self.model.core.last_layer
        else:
            proto_class_identity = self.model.prototype_class_identity
            last_layer = self.model.last_layer
        return proto_class_identity, last_layer

    def _apply_ema_update(self):
        """Apply exponential moving average to model parameters."""
        if self.ema_state is None:
            self.ema_state = deepcopy(self.model.state_dict())
        else:
            current_state = self.model.state_dict()
            for key in self.ema_state:
                if current_state[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    self.ema_state[key] = (self.ema_alpha * self.ema_state[key] + 
                                          (1 - self.ema_alpha) * current_state[key])
            self.model.load_state_dict(self.ema_state, strict=True)

    def _update_ema_loss(self, loss_value):
        """Update EMA of loss for model recovery."""
        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss_value

    def _check_model_recovery(self):
        """Check if model should be reset (SAR-style recovery)."""
        if self.use_model_recovery and self.ema_loss is not None:
            if self.ema_loss < self.recovery_threshold:
                print(f"EMA loss {self.ema_loss:.3f} < {self.recovery_threshold}, resetting model")
                self.reset()
                self.adaptation_stats['model_resets'] += 1
                return True
        return False

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward pass with enhanced prototype-aware entropy minimization.
        
        Improvements:
        1. SAR-style entropy filtering
        2. Optional SAM double-pass
        3. Hybrid loss (proto + softmax entropy)
        4. Adaptive thresholds
        """
        if self.use_sam:
            return self._forward_and_adapt_sam(x)
        else:
            return self._forward_and_adapt_standard(x)

    def _forward_and_adapt_sam(self, x):
        """Forward with Sharpness-Aware Minimization (double pass)."""
        # First forward pass
        self.optimizer.zero_grad()
        outputs = self.model(x)
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, min_distances = outputs[0], outputs[1]
        else:
            logits = outputs
            min_distances = None
        
        if min_distances is None:
            # Fallback to simple softmax entropy
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            
            outputs_2 = self.model(x)
            logits_2 = outputs_2[0] if isinstance(outputs_2, tuple) else outputs_2
            loss_2 = softmax_entropy(logits_2).mean()
            loss_2.backward()
            self.optimizer.second_step(zero_grad=True)
            return outputs
        
        # Compute loss at current weights
        loss_1, reliable_mask = self._compute_loss(logits, min_distances, return_mask=True)
        
        if loss_1.item() == 0 or reliable_mask.sum() == 0:
            return outputs
        
        loss_1.backward()
        
        # SAM first step: move to w + epsilon
        self.optimizer.first_step(zero_grad=True)
        
        # Second forward pass at perturbed weights
        outputs_2 = self.model(x)
        if isinstance(outputs_2, tuple) and len(outputs_2) >= 2:
            logits_2, min_distances_2 = outputs_2[0], outputs_2[1]
        else:
            logits_2 = outputs_2
            min_distances_2 = None
        
        # Re-compute loss at perturbed weights (use same mask for consistency)
        if min_distances_2 is not None:
            loss_2, _ = self._compute_loss(logits_2, min_distances_2, return_mask=True, cached_mask=reliable_mask)
        else:
            loss_2 = softmax_entropy(logits_2).mean()
        
        if loss_2.item() > 0:
            loss_2.backward()
            self.optimizer.second_step(zero_grad=True)
            
            # Track adapted samples
            self.adaptation_stats['adapted_samples'] += int(reliable_mask.sum().item())
            self.adaptation_stats['total_updates'] += 1
            
            # Update EMA loss for recovery
            self._update_ema_loss(loss_2.item())
            self._check_model_recovery()
        
        return outputs

    def _forward_and_adapt_standard(self, x):
        """Standard single-pass forward and adapt."""
        outputs = self.model(x)
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, min_distances = outputs[0], outputs[1]
        else:
            logits = outputs
            min_distances = None
            
        if min_distances is None:
            # Fallback to simple softmax entropy
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return outputs
        
        loss, reliable_mask = self._compute_loss(logits, min_distances, return_mask=True)
        
        if loss.item() > 0 and reliable_mask.sum() > 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Track adapted samples
            self.adaptation_stats['adapted_samples'] += int(reliable_mask.sum().item())
            self.adaptation_stats['total_updates'] += 1
        
        return outputs

    def _compute_loss(self, logits, min_distances, return_mask=False, cached_mask=None):
        """Compute the hybrid loss with filtering.
        
        Returns:
            loss: Combined loss value
            reliable_mask: (optional) Mask of reliable samples
        """
        device = logits.device
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Convert distances to similarities
        raw_similarities = torch.log((min_distances + 1.0) / (min_distances + 1e-4))  # [B, P]
        similarities = raw_similarities / 9.0  # Normalize to roughly [0, 1]
        
        # Get prototype class identity
        proto_class_identity, last_layer = self._get_model_components()
        
        with torch.no_grad():
            pred_class = logits.argmax(dim=1)  # [B]
            proto_identities = proto_class_identity.argmax(dim=1).to(device)  # [P]
        
        # ============ FILTERING ============
        if cached_mask is not None:
            reliable_mask = cached_mask
        else:
            reliable_mask = torch.ones(batch_size, device=device)
            
            # 1. Entropy filtering (SAR-style)
            if self.use_entropy_filter:
                margin_e0 = self.entropy_margin_scale * math.log(num_classes)
                entropy = softmax_entropy(logits)
                entropy_mask = (entropy < margin_e0).float()
                reliable_mask = reliable_mask * entropy_mask
                self.adaptation_stats['filtered_by_entropy'] += int((1 - entropy_mask).sum().item())
            
            # 2. Geometric filtering
            if self.use_geometric_filter:
                max_sim_per_sample = similarities.max(dim=1)[0]  # [B]
                
                # Adaptive threshold based on batch statistics
                if self.use_adaptive_threshold:
                    batch_mean = max_sim_per_sample.mean()
                    batch_std = max_sim_per_sample.std()
                    adaptive_thresh = max(0.1, batch_mean - batch_std)  # At least 0.1
                    geo_mask = (max_sim_per_sample > adaptive_thresh).float()
                else:
                    geo_mask = (max_sim_per_sample > self.geo_filter_threshold).float()
                
                reliable_mask = reliable_mask * geo_mask
                self.adaptation_stats['filtered_by_geo'] += int((1 - geo_mask).sum().item())
                
                # Track geo stats
                self.geo_filter_stats['total_samples'] += batch_size
                self.geo_filter_stats['filtered_samples'] += int((1 - geo_mask).sum().item())
        
        if reliable_mask.sum() == 0:
            if return_mask:
                return torch.tensor(0.0, device=device), reliable_mask
            return torch.tensor(0.0, device=device)
        
        sample_weights = reliable_mask.unsqueeze(1)  # [B, 1]
        
        # ============ LOSS COMPUTATION ============
        
        # Create target/non-target masks
        if self.adapt_all_prototypes:
            num_prototypes = proto_identities.shape[0]
            target_mask = torch.ones(batch_size, num_prototypes, device=device)
        else:
            target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        nontarget_mask = 1.0 - target_mask
        
        loss = torch.tensor(0.0, device=device)
        
        # ========== Part A: Binary Prototype Entropy ==========
        if self.alpha_proto > 0:
            eps = 1e-6
            masked_sims = similarities * target_mask
            masked_sims = torch.clamp(masked_sims, min=0.0, max=1.0)
            proto_probs = torch.clamp(masked_sims, min=eps, max=1-eps)
            
            # Binary entropy
            entropy = -(proto_probs * torch.log(proto_probs) + 
                       (1 - proto_probs) * torch.log(1 - proto_probs))
            
            # Apply importance weighting
            if self.use_prototype_importance:
                class_weights = last_layer.weight[pred_class]  # [B, P]
                importance_weights = torch.abs(class_weights) * target_mask
                importance_weights = importance_weights / (importance_weights.sum(dim=1, keepdim=True) + 1e-8)
                weighted_entropy = entropy * importance_weights * sample_weights
                loss_per_sample = weighted_entropy.sum(dim=1)
            else:
                masked_entropy = entropy * target_mask * sample_weights
                loss_per_sample = masked_entropy.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)
            
            # Confidence weighting
            if self.use_confidence_weighting:
                with torch.no_grad():
                    confidence = logits.softmax(dim=1).max(dim=1)[0]
                loss_proto = (loss_per_sample * confidence * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
            else:
                loss_proto = (loss_per_sample * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
            
            loss = loss + self.alpha_proto * loss_proto
        
        # ========== Part B: Softmax Entropy (SAR-style) ==========
        if self.alpha_softmax > 0:
            softmax_ent = softmax_entropy(logits)  # [B]
            loss_softmax = (softmax_ent * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
            loss = loss + self.alpha_softmax * loss_softmax
        
        # ========== Part C: Separation Loss ==========
        if self.alpha_separation > 0:
            nontarget_sims = similarities * nontarget_mask
            nontarget_sims = torch.clamp(nontarget_sims, min=0.0, max=1.0)
            eps = 1e-6
            separation_loss = -torch.log(1 - nontarget_sims + eps) * nontarget_mask * sample_weights
            loss_sep_per_sample = separation_loss.sum(dim=1) / (nontarget_mask.sum(dim=1) + 1e-8)
            loss_separation = (loss_sep_per_sample * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
            loss = loss + self.alpha_separation * loss_separation
        
        if return_mask:
            return loss, reliable_mask
        return loss

    def reset(self):
        """Reset model parameters to pretrained state."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.ema_state = None
        self.ema_loss = None
        
    def get_stats(self):
        """Get adaptation statistics."""
        return {
            **self.adaptation_stats,
            'geo_filter_stats': self.geo_filter_stats.copy()
        }


# ============================================================================
# Setup Functions
# ============================================================================

def collect_params_enhanced(model, adaptation_mode='batchnorm_addon'):
    """Collect parameters for enhanced ProtoEntropy."""
    params = []
    names = []
    
    if hasattr(model, 'core'):
        core = model.core
        prefix = 'core.'
    else:
        core = model
        prefix = ''
    
    # BatchNorm params
    if 'batchnorm' in adaptation_mode or adaptation_mode == 'all_adapt':
        for nm, m in core.features.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}features.{nm}.{np_name}")
        
        for nm, m in core.add_on_layers.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}add_on_layers.{nm}.{np_name}")
    
    # Add-on layer params
    if 'addon' in adaptation_mode or adaptation_mode == 'all_adapt':
        for nm, p in core.add_on_layers.named_parameters():
            param_name = f"{prefix}add_on_layers.{nm}"
            if p.requires_grad and param_name not in names:
                params.append(p)
                names.append(param_name)
    
    # Prototype vectors
    if 'proto' in adaptation_mode or adaptation_mode == 'all_adapt':
        if hasattr(core, 'prototype_vectors') and core.prototype_vectors.requires_grad:
            params.append(core.prototype_vectors)
            names.append(f"{prefix}prototype_vectors")
    
    return params, names


def configure_model_enhanced(model, adaptation_mode='batchnorm_addon'):
    """Configure model for enhanced ProtoEntropy."""
    if hasattr(model, 'core'):
        core = model.core
    else:
        core = model
    
    model.train()
    model.requires_grad_(False)
    
    if 'batchnorm' in adaptation_mode or adaptation_mode == 'all_adapt':
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
    
    if 'addon' in adaptation_mode or adaptation_mode == 'all_adapt':
        for p in core.add_on_layers.parameters():
            p.requires_grad = True
    
    if 'proto' in adaptation_mode or adaptation_mode == 'all_adapt':
        if hasattr(core, 'prototype_vectors'):
            core.prototype_vectors.requires_grad = True
    
    return model


def setup_proto_entropy_enhanced(model, 
                                  lr=0.001,
                                  use_sam=False,
                                  alpha_proto=1.0,
                                  alpha_softmax=0.0,
                                  use_entropy_filter=True,
                                  entropy_margin_scale=0.4,
                                  use_geometric_filter=False,
                                  geo_filter_threshold=0.3,
                                  use_adaptive_threshold=False,
                                  adaptation_mode='batchnorm_addon',
                                  use_ema=False,
                                  use_model_recovery=False,
                                  steps=1):
    """
    Set up enhanced ProtoEntropy adaptation.
    
    Recommended configurations:
    
    1. ProtoTTA-SAM (best for blur):
       use_sam=True, alpha_proto=0.5, alpha_softmax=0.5
    
    2. ProtoTTA-Hybrid (balanced):
       alpha_proto=0.7, alpha_softmax=0.3, use_entropy_filter=True
    
    3. ProtoTTA-Adaptive (auto-tuning):
       use_adaptive_threshold=True, use_ema=True
    """
    model = configure_model_enhanced(model, adaptation_mode=adaptation_mode)
    params, param_names = collect_params_enhanced(model, adaptation_mode=adaptation_mode)
    
    if not params:
        print(f"Warning: No parameters found for mode {adaptation_mode}")
        return model
    
    # Create optimizer
    if use_sam:
        optimizer = SAM(params, torch.optim.SGD, lr=lr, momentum=0.9, rho=0.05)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    
    proto_model = ProtoEntropyEnhanced(
        model,
        optimizer,
        steps=steps,
        alpha_proto=alpha_proto,
        alpha_softmax=alpha_softmax,
        use_entropy_filter=use_entropy_filter,
        entropy_margin_scale=entropy_margin_scale,
        use_geometric_filter=use_geometric_filter,
        geo_filter_threshold=geo_filter_threshold,
        use_adaptive_threshold=use_adaptive_threshold,
        use_sam=use_sam,
        use_ema=use_ema,
        use_model_recovery=use_model_recovery,
    )
    
    mode_str = []
    if use_sam:
        mode_str.append("SAM")
    if use_entropy_filter:
        mode_str.append(f"ent@{entropy_margin_scale}")
    if use_geometric_filter:
        mode_str.append(f"geo@{geo_filter_threshold}")
    if use_adaptive_threshold:
        mode_str.append("adaptive")
    if use_ema:
        mode_str.append("EMA")
    if alpha_softmax > 0:
        mode_str.append(f"hybrid({alpha_proto:.1f}:{alpha_softmax:.1f})")
    
    print(f"ProtoEntropy++ ({'+'.join(mode_str) if mode_str else 'basic'}, {adaptation_mode}, {len(params)} params)")
    
    return proto_model
