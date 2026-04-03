"""
Prototype-aware Entropy Minimization (ProtoEntropy) for ProtoPNet.

This module implements prototype-aware test-time adaptation that leverages
the interpretable structure of prototype networks.

Key difference from Tent/EATA:
- Uses BINARY ENTROPY of prototype similarities (not softmax entropy of logits)
- Encourages each prototype to be clearly ON or OFF for each sample
- Supports importance weighting from last_layer weights

Adapted from the ProtoViT TTA framework for CNN-based ProtoPNet.
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for reset capability."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict()) if optimizer else None
    return model_state, optimizer_state


class ProtoEntropy(nn.Module):
    """Prototype-aware entropy minimization for test-time adaptation.

    Uses binary entropy of prototype similarities (not softmax entropy of logits).
    This is the key difference from Tent and EATA.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 alpha_target=1.0, alpha_separation=0.0,
                 use_prototype_importance=False,
                 use_confidence_weighting=False,
                 reset_mode=None,
                 reset_frequency=10,
                 confidence_threshold=0.7,
                 use_geometric_filter=False,
                 geo_filter_threshold=0.3,
                 use_ensemble_entropy=False,
                 source_proto_stats=None,
                 alpha_source_kl=0.0,
                 adapt_all_prototypes=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ProtoEntropy requires >= 1 step(s)"
        self.episodic = episodic

        # Loss weights
        self.alpha_target = alpha_target
        self.alpha_separation = alpha_separation
        
        # Prototype-specific settings
        self.use_prototype_importance = use_prototype_importance
        self.use_confidence_weighting = use_confidence_weighting
        
        # Reset strategies
        if reset_mode is None:
            self.reset_mode = 'episodic' if episodic else 'none'
        else:
            self.reset_mode = reset_mode
        self.reset_frequency = reset_frequency
        self.confidence_threshold = confidence_threshold
        
        # Geometric filtering
        self.use_geometric_filter = use_geometric_filter
        self.geo_filter_threshold = geo_filter_threshold
        
        # Ensemble entropy
        self.use_ensemble_entropy = use_ensemble_entropy
        
        # Source statistics for KL regularization
        self.source_proto_stats = source_proto_stats
        self.alpha_source_kl = alpha_source_kl
        
        # Whether to adapt all prototypes or just predicted class
        self.adapt_all_prototypes = adapt_all_prototypes

        # Save model/optimizer state for reset
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Tracking
        self.batch_count = 0
        self.geo_filter_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'total_updates': 0,
            'min_similarities': [],
            'max_similarities': [],
            'avg_similarities': []
        }

    def forward(self, x):
        if self.reset_mode == 'episodic':
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        
        self.batch_count += 1
        
        # Handle periodic reset
        if self.reset_mode == 'periodic' and self.batch_count % self.reset_frequency == 0:
            self.reset()
        
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

    def _distance_to_similarity(self, min_distances):
        """Convert ProtoPNet distances to similarities in range [-1, 1].
        
        ProtoPNet uses log activation: sim = log((d+1)/(d+eps))
        This gives range [0, ~9.2]. We normalize to [-1, 1] for consistency
        with ProtoViT's cosine similarities.
        """
        # Raw log similarity (range: 0 to ~9.2)
        raw_sim = torch.log((min_distances + 1.0) / (min_distances + 1e-4))
        
        # Normalize to [-1, 1] using tanh-like scaling
        # The value 5.0 is chosen so that raw_sim=0 -> -0.8, raw_sim=9.2 -> 0.8
        normalized_sim = 2 * torch.sigmoid(raw_sim - 2.0) - 1
        
        return normalized_sim

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward pass with prototype-aware binary entropy minimization.
        
        This closely mirrors ProtoViT's forward_and_adapt logic:
        1. Forward pass to get logits and min_distances
        2. Convert distances to similarities in [-1, 1]
        3. Create target/non-target masks based on predicted class
        4. Compute BINARY ENTROPY of prototype similarities
        5. Apply importance weighting, confidence weighting, geometric filtering
        6. Backward and update
        """
        # Forward through model
        outputs = self.model(x)
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, min_distances = outputs[0], outputs[1]
        else:
            logits = outputs
            min_distances = None
            
        if min_distances is None:
            # Fallback to simple entropy if no distances available
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return outputs
        
        # Convert distances to similarities
        # Use ProtoPNet's standard log activation directly (range ~0 to ~9)
        # Don't normalize to [-1,1] as it crushes variance
        raw_similarities = torch.log((min_distances + 1.0) / (min_distances + 1e-4))  # [B, P]
        
        # Normalize by a reasonable scale for filtering
        # Typical range is 0-9, so normalize to roughly 0-1 for thresholding
        similarities = raw_similarities / 9.0  # [B, P]
        
        # Get prototype class identity and last layer
        proto_class_identity, last_layer = self._get_model_components()
        
        # Identify Target Class (Pseudo-label)
        with torch.no_grad():
            pred_class = logits.argmax(dim=1)  # [B]
            proto_identities = proto_class_identity.argmax(dim=1).to(logits.device)  # [P]
        
        # Create target/non-target masks
        if self.adapt_all_prototypes:
            batch_size = logits.shape[0]
            num_prototypes = proto_identities.shape[0]
            target_mask = torch.ones(batch_size, num_prototypes, 
                                    device=logits.device, dtype=torch.float32)
        else:
            target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        nontarget_mask = 1.0 - target_mask
        
        # ========== Geometric Filtering: Filter unreliable samples ==========
        if self.use_geometric_filter:
            with torch.no_grad():
                max_sim_per_sample = similarities.max(dim=1)[0]  # [B]
                max_raw_sim = raw_similarities.max(dim=1)[0]  # [B] - for logging
                
                # Track statistics (both normalized and raw for diagnostics)
                batch_size = max_sim_per_sample.shape[0]
                self.geo_filter_stats['total_samples'] += batch_size
                num_filtered = (max_sim_per_sample <= self.geo_filter_threshold).sum().item()
                self.geo_filter_stats['filtered_samples'] += num_filtered
                self.geo_filter_stats['min_similarities'].append(max_sim_per_sample.min().item())
                self.geo_filter_stats['max_similarities'].append(max_sim_per_sample.max().item())
                self.geo_filter_stats['avg_similarities'].append(max_sim_per_sample.mean().item())
                
                # Also track raw similarities for diagnostics
                if 'raw_min_similarities' not in self.geo_filter_stats:
                    self.geo_filter_stats['raw_min_similarities'] = []
                    self.geo_filter_stats['raw_max_similarities'] = []
                    self.geo_filter_stats['raw_avg_similarities'] = []
                self.geo_filter_stats['raw_min_similarities'].append(max_raw_sim.min().item())
                self.geo_filter_stats['raw_max_similarities'].append(max_raw_sim.max().item())
                self.geo_filter_stats['raw_avg_similarities'].append(max_raw_sim.mean().item())
                
                reliable_mask = (max_sim_per_sample > self.geo_filter_threshold).float()
                
                if reliable_mask.sum() == 0:
                    return outputs
                
                # Track number of adapted samples (those that passed the filter)
                num_adapted = int(reliable_mask.sum().item())
                self.geo_filter_stats['total_updates'] += num_adapted
        else:
            reliable_mask = torch.ones(logits.shape[0], device=logits.device)
            # Track all samples as adapted when no filtering
            num_adapted = logits.shape[0]
            self.geo_filter_stats['total_updates'] += num_adapted
        
        sample_weights = reliable_mask.unsqueeze(1)  # [B, 1]
        
        # ========== PART A: Target Entropy Loss (Binary Entropy) ==========
        eps = 1e-6
        
        # Apply target mask to similarities
        masked_sims = similarities * target_mask
        masked_sims = torch.clamp(masked_sims, min=0.0, max=1.0)  # Similarities are now [0, 1] range
        
        # Use similarities directly as probabilities (already in [0, 1] range)
        proto_probs = masked_sims  # [B, P]
        proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)
        
        # Binary Entropy: -(p log p + (1-p) log (1-p))
        # Minimizing this encourages each prototype to be clearly ON (p=1) or OFF (p=0)
        entropy = -(proto_probs * torch.log(proto_probs) + 
                   (1 - proto_probs) * torch.log(1 - proto_probs))  # [B, P]
        
        # --- Prototype Importance Weighting ---
        if self.use_prototype_importance:
            last_layer_weights = last_layer.weight  # [num_classes, num_prototypes]
            class_weights = last_layer_weights[pred_class]  # [B, P]
            
            # Normalize weights
            importance_weights = torch.abs(class_weights)
            importance_weights = importance_weights * target_mask
            importance_weights = importance_weights / (importance_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weight entropy by importance and sample reliability
            weighted_entropy = entropy * target_mask * importance_weights * sample_weights
            loss_per_sample = weighted_entropy.sum(dim=1)  # [B]
        else:
            # Uniform weighting across target prototypes
            masked_entropy = entropy * target_mask * sample_weights
            loss_per_sample = masked_entropy.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)  # [B]
        
        # --- Confidence Weighting ---
        if self.use_confidence_weighting:
            with torch.no_grad():
                probs = logits.softmax(dim=1)
                confidence = probs.max(dim=1)[0]  # [B]
            
            loss_target = (loss_per_sample * confidence * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
        else:
            loss_target = (loss_per_sample * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)

        # ========== PART B: Separation Loss (push non-target protos away) ==========
        loss_separation = 0.0
        if self.alpha_separation > 0:
            nontarget_sims = similarities * nontarget_mask
            nontarget_sims = torch.clamp(nontarget_sims, min=0.0, max=1.0)  # Already in [0, 1]
            nontarget_probs = nontarget_sims
            nontarget_probs = torch.clamp(nontarget_probs, min=eps, max=1-eps)
            
            # Push non-target probabilities toward 0 (low similarity)
            separation_loss = -torch.log(1 - nontarget_probs + eps) * nontarget_mask * sample_weights
            loss_sep_per_sample = separation_loss.sum(dim=1) / (nontarget_mask.sum(dim=1) + 1e-8)
            loss_separation = (loss_sep_per_sample * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
        
        # ========== Total Loss ==========
        loss = self.alpha_target * loss_target + self.alpha_separation * loss_separation

        # Backward and update
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return outputs

    def reset(self):
        """Reset model parameters to pretrained state."""
        self.model.load_state_dict(self.model_state, strict=True)
        
    def get_geo_filter_stats(self):
        """Get geometric filtering statistics."""
        stats = self.geo_filter_stats.copy()
        if stats['total_samples'] > 0:
            stats['filter_rate'] = stats['filtered_samples'] / stats['total_samples']
        else:
            stats['filter_rate'] = 0.0
        return stats


class ProtoEntropyEATA(nn.Module):
    """ProtoEntropy with EATA-style entropy thresholding.
    
    Combines prototype binary entropy with sample filtering.
    Only adapts on samples with low softmax entropy (high confidence).
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_threshold=0.4):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.entropy_threshold = entropy_threshold
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
    
    def _get_model_components(self):
        if hasattr(self.model, 'core'):
            proto_class_identity = self.model.core.prototype_class_identity
            last_layer = self.model.core.last_layer
        else:
            proto_class_identity = self.model.prototype_class_identity
            last_layer = self.model.last_layer
        return proto_class_identity, last_layer

    def _distance_to_similarity(self, min_distances):
        raw_sim = torch.log((min_distances + 1.0) / (min_distances + 1e-4))
        normalized_sim = 2 * torch.sigmoid(raw_sim - 2.0) - 1
        return normalized_sim

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        outputs = self.model(x)
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, min_distances = outputs[0], outputs[1]
        else:
            logits = outputs
            min_distances = None
            
        if min_distances is None:
            return outputs
        
        similarities = self._distance_to_similarity(min_distances)
        proto_class_identity, _ = self._get_model_components()

        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            
            # EATA-style entropy thresholding on logits
            num_classes = logits.shape[1]
            adaptive_threshold = self.entropy_threshold * torch.log(
                torch.tensor(num_classes, device=logits.device).float()
            )
            
            softmax_ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
            reliable_mask = (softmax_ent < adaptive_threshold).float()
            
            if reliable_mask.sum() == 0:
                return outputs
            
            proto_identities = proto_class_identity.argmax(dim=1).to(logits.device)

        # Target mask
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        sample_weights = reliable_mask.unsqueeze(1)

        # Binary entropy on prototype probabilities
        eps = 1e-6
        masked_sims = similarities * target_mask
        masked_sims = torch.clamp(masked_sims, min=-1.0, max=1.0)
        proto_probs = (masked_sims + 1.0) / 2.0
        proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)
        
        entropy = -(proto_probs * torch.log(proto_probs) + 
                   (1 - proto_probs) * torch.log(1 - proto_probs))
        
        target_loss_map = entropy * target_mask * sample_weights
        loss_per_sample = target_loss_map.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)
        loss = (loss_per_sample * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return outputs


# ============================================================================
# Parameter Collection and Model Configuration
# ============================================================================

def collect_params(model, adaptation_mode='all_adapt'):
    """
    Collect parameters for adaptation based on mode.
    
    Following ProtoViT approach: include BatchNorm for better TTA performance.
    
    For VGG-based ProtoPNet, we can adapt:
    - 'batchnorm_only': Only BatchNorm layers
    - 'addon_only': Only Add-on layer Conv weights
    - 'proto_only': Only prototype vectors
    - 'addon_proto': Add-on layers + prototype vectors
    - 'batchnorm_addon': BatchNorm + Add-on layers (recommended)
    - 'batchnorm_proto': BatchNorm + prototype vectors
    - 'all_adapt': BatchNorm + Add-on layers + prototype vectors (default)
    """
    params = []
    names = []
    
    if hasattr(model, 'core'):
        core = model.core
        prefix = 'core.'
    else:
        core = model
        prefix = ''
    
    # Collect BatchNorm params (for vgg*_bn models)
    # Include BatchNorm by default for better adaptation (following ProtoViT)
    if 'batchnorm' in adaptation_mode or adaptation_mode == 'all_adapt':
        # Collect from features (VGG backbone)
        for nm, m in core.features.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}features.{nm}.{np_name}")
        
        # Collect from add_on_layers (if they have BatchNorm)
        for nm, m in core.add_on_layers.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}add_on_layers.{nm}.{np_name}")
    
    # Collect Add-on layer Conv params
    if 'addon' in adaptation_mode or adaptation_mode == 'all_adapt':
        for nm, p in core.add_on_layers.named_parameters():
            param_name = f"{prefix}add_on_layers.{nm}"
            if p.requires_grad and param_name not in names:
                params.append(p)
                names.append(param_name)
    
    # Collect prototype vectors
    if 'proto' in adaptation_mode or adaptation_mode == 'all_adapt':
        if hasattr(core, 'prototype_vectors') and core.prototype_vectors.requires_grad:
            params.append(core.prototype_vectors)
            names.append(f"{prefix}prototype_vectors")
    
    return params, names


def configure_model(model, adaptation_mode='all_adapt'):
    """Configure model for ProtoEntropy adaptation.
    
    Following ProtoViT: Use train mode for BatchNorm with track_running_stats=False
    to force batch statistics. This gives better TTA performance.
    """
    # Get the core model
    if hasattr(model, 'core'):
        core = model.core
    else:
        core = model
    
    # Put model in train mode for BatchNorm adaptation
    model.train()
    
    # Disable all gradients initially
    model.requires_grad_(False)
    
    # Configure BatchNorm layers
    if 'batchnorm' in adaptation_mode or adaptation_mode == 'all_adapt':
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.requires_grad_(True)
                # Force use of batch statistics instead of running statistics
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
    
    # Enable add_on_layers
    if 'addon' in adaptation_mode or adaptation_mode == 'all_adapt':
        for p in core.add_on_layers.parameters():
            p.requires_grad = True
    
    # Enable prototype vectors
    if 'proto' in adaptation_mode or adaptation_mode == 'all_adapt':
        if hasattr(core, 'prototype_vectors'):
            core.prototype_vectors.requires_grad = True
    
    # Get parameter count for logging
    params_to_enable, param_names = collect_params(model, adaptation_mode)
    print(f"ProtoEntropy configured with {len(params_to_enable)} adaptable parameters: {param_names[:5]}{'...' if len(param_names) > 5 else ''}")
    
    return model
