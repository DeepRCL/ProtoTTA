from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def relative_evidence_lambda(logits, similarities, prototype_classes,
                             topk=3, eps=1e-8):
    """Relative prototype-vs-output class evidence for one test batch."""
    num_classes = logits.shape[1]
    counts = torch.bincount(prototype_classes, minlength=num_classes)
    order = torch.argsort(prototype_classes)

    # ProtoViT has an equal number of prototypes per class. Use a grouped
    # tensor in that common case to avoid one GPU operation per class.
    if counts.min() > 0 and torch.equal(counts, counts[0].expand_as(counts)):
        per_class = similarities[:, order].reshape(
            similarities.shape[0], num_classes, int(counts[0].item())
        )
        k = min(max(int(topk), 1), per_class.shape[2])
        class_scores = per_class.topk(k, dim=2).values.mean(dim=2)
    else:
        grouped = []
        for class_index in range(num_classes):
            class_mask = prototype_classes == class_index
            if not class_mask.any():
                grouped.append(similarities.new_full(
                    (similarities.shape[0],), float('-inf')
                ))
                continue
            per_class = similarities[:, class_mask]
            k = min(max(int(topk), 1), per_class.shape[1])
            grouped.append(per_class.topk(k, dim=1).values.mean(dim=1))
        class_scores = torch.stack(grouped, dim=1)

    pred_class = logits.argmax(dim=1)
    selected = class_scores.gather(1, pred_class.unsqueeze(1)).squeeze(1)
    competitor = class_scores.scatter(
        1, pred_class.unsqueeze(1), float('-inf')
    ).max(dim=1).values
    proto_reliability = (
        (selected - competitor).clamp_min(0.0)
        / selected.abs().clamp_min(eps)
    ).clamp(0.0, 1.0)

    top2 = logits.softmax(dim=1).topk(2, dim=1).values
    output_reliability = (
        (top2[:, 0] - top2[:, 1]) / top2[:, 0].clamp_min(eps)
    ).clamp(0.0, 1.0)
    coefficient = proto_reliability / (
        proto_reliability + output_reliability + eps
    )
    return coefficient.detach(), proto_reliability, output_reliability

class ProtoEntropy(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 alpha_target=1.0, alpha_separation=0.0, alpha_coherence=0.0,
                 use_prototype_importance=False, use_confidence_weighting=False,
                 reset_mode=None, reset_frequency=10,
                 confidence_threshold=0.7, ema_alpha=0.999,
                 use_geometric_filter=False, geo_filter_threshold=0.3,
                 consensus_strategy='max', consensus_ratio=0.5,
                 use_ensemble_entropy=False,
                 source_proto_stats=None, alpha_source_kl=0.0,
                 adapt_all_prototypes=False,
                 logit_weight=0.0,
                 shared_confidence_weighting=False,
                 gradient_normalize=False,
                 adaptive_lambda=False,
                 adaptive_lambda_strategy='relative_reliability',
                 adaptive_delta0=0.25,
                 adaptive_topk=3,
                 router_min_consistency=0.25,
                 lambda_ema_momentum=0.9,
                 lambda_min=0.05,
                 lambda_max=0.95,
                 record_diagnostics=False,
                 samplewise_lambda=False,
                 semantic_prototype_weights=None,
                 compatibility_student=None,
                 compatibility_floor=0.25,
                 semantic_logit_blend=0.0,
                 semantic_fusion='fixed',
                 semantic_contrast_weight=0.0,
                 semantic_temperature=0.25):
        """
        Args:
            episodic: If True, uses 'episodic' reset_mode (backward compatibility)
            reset_mode: 'episodic' (reset every sample), 'periodic' (reset every N batches),
                       'confidence' (reset when confidence drops), 'ema' (exponential moving average),
                       'none' (no reset), 'hybrid' (combine periodic + confidence)
                       If None, infers from episodic flag: True='episodic', False='none'
            reset_frequency: How often to reset in 'periodic' mode in BATCHES (e.g., 10 = every 10 batches)
                           With batch_size=128, reset_frequency=10 means reset every 1280 samples
            confidence_threshold: Minimum avg confidence before triggering reset in 'confidence' mode
            ema_alpha: EMA decay factor for 'ema' mode (closer to 1 = slower adaptation)
            use_geometric_filter: If True, filter unreliable samples using geometric similarity to prototypes
            geo_filter_threshold: Minimum similarity to ANY prototype to be considered reliable (default: 0.3)
                                 Higher = more strict filtering, Lower = accept more samples
            consensus_strategy: How to aggregate sub-prototype similarities for filtering/adaptation:
                               'max' - use best sub-prototype (original, can be fooled by outliers)
                               'mean' - use average across sub-prototypes (requires consensus)
                               'median' - use median across sub-prototypes (robust to outliers)
                               'top_k_mean' - average of top-K sub-prototypes (soft consensus)
                               'weighted_mean' - similarity-weighted mean (high sims matter more)
            consensus_ratio: For 'top_k_mean', what fraction of sub-prototypes to use (default: 0.5 = top 50%)
            use_ensemble_entropy: If True, treat sub-prototypes as ensemble: compute entropy per sub-proto, then average
                                 If False, aggregate sub-protos first, then compute entropy (default)
            source_proto_stats: Dict with source prototype activation statistics (mean, std, distribution)
                               Computed on clean/source data before adaptation
            alpha_source_kl: Weight for KL divergence regularization to source distribution (default: 0.0 = disabled)
            adapt_all_prototypes: If True, adapt based on all prototypes (not just target prototypes matching predicted class)
                                 If False (default), only adapt target prototypes (original behavior)
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic  # Keep for backward compatibility
        self.alpha_target = alpha_target
        self.alpha_separation = alpha_separation
        self.alpha_coherence = alpha_coherence
        self.use_prototype_importance = use_prototype_importance
        self.use_confidence_weighting = use_confidence_weighting
        
        # Geometric filtering parameters
        self.use_geometric_filter = use_geometric_filter
        self.geo_filter_threshold = geo_filter_threshold
        
        # Consensus strategy parameters
        self.consensus_strategy = consensus_strategy
        self.consensus_ratio = consensus_ratio
        
        # Ensemble entropy parameter
        self.use_ensemble_entropy = use_ensemble_entropy
        
        # Source distribution matching parameters
        self.source_proto_stats = source_proto_stats
        self.alpha_source_kl = alpha_source_kl
        
        # Adaptation scope parameter
        self.adapt_all_prototypes = adapt_all_prototypes

        # Unified λ: fraction of logit-entropy in the total loss (0 = pure proto, 1 = pure logit)
        self.logit_weight = logit_weight
        self.shared_confidence_weighting = shared_confidence_weighting
        self.gradient_normalize = gradient_normalize
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_lambda_strategy = adaptive_lambda_strategy
        self.adaptive_delta0 = adaptive_delta0
        self.adaptive_topk = adaptive_topk
        self.router_min_consistency = router_min_consistency
        self.lambda_ema_momentum = lambda_ema_momentum
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.record_diagnostics = record_diagnostics
        self.samplewise_lambda = samplewise_lambda
        self.lambda_ema = None
        self.compatibility_student = compatibility_student
        self.compatibility_floor = float(compatibility_floor)
        self.semantic_logit_blend = float(semantic_logit_blend)
        if semantic_fusion not in ('fixed', 'confidence_guard'):
            raise ValueError(f'Unknown semantic fusion strategy: {semantic_fusion}')
        self.semantic_fusion = semantic_fusion
        self.semantic_contrast_weight = float(semantic_contrast_weight)
        self.semantic_temperature = float(semantic_temperature)
        if semantic_prototype_weights is None:
            semantic_prototype_weights = torch.ones(
                model.prototype_class_identity.shape[0], dtype=torch.float32
            )
            self.use_semantic_prototype_weights = False
        else:
            semantic_prototype_weights = torch.as_tensor(
                semantic_prototype_weights, dtype=torch.float32
            )
            self.use_semantic_prototype_weights = True
        self.register_buffer(
            'semantic_prototype_weights', semantic_prototype_weights.clamp(0.05, 1.0)
        )
        
        # New reset mechanism parameters
        # If reset_mode not specified, infer from episodic flag for backward compatibility
        if reset_mode is None:
            self.reset_mode = 'episodic' if episodic else 'none'
        else:
            self.reset_mode = reset_mode
        
        self.reset_frequency = reset_frequency
        self.confidence_threshold = confidence_threshold
        self.ema_alpha = ema_alpha
        
        # State tracking
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.batch_count = 0  # Track number of batches, not samples
        self.sample_count = 0  # Keep for reference
        self.confidence_history = []
        self.ema_state = None
        
        # Statistics tracking for geometric filtering
        self.geo_filter_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'min_similarities': [],  # Track min similarity per batch
            'max_similarities': [],  # Track max similarity per batch
            'avg_similarities': []   # Track avg similarity per batch
        }
        
        # Adaptation tracking statistics (for comparison with other TTA methods)
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,  # Samples that passed filtering
            'total_updates': 0,    # Total optimizer steps
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

    def forward(self, x):
        # Handle different reset strategies
        should_reset = self._check_reset_condition(x)
        if should_reset:
            self.reset()
        
        # Track adaptation
        batch_size = x.size(0)
        self.adaptation_stats['total_samples'] += batch_size
        
        for _ in range(self.steps):
            outputs, min_distances, similarities = self.forward_and_adapt(x)

        # Update tracking after adaptation
        self._update_tracking(outputs)
        
        # EMA update if in EMA mode
        if self.reset_mode == 'ema':
            self._ema_update()

        return outputs, min_distances, similarities

    def _check_reset_condition(self, x):
        """Determine if we should reset based on the reset_mode.
        
        Note: reset_frequency is now in terms of BATCHES, not individual samples.
        E.g., reset_frequency=10 means reset every 10 batches.
        """
        if self.reset_mode == 'episodic':
            return True
        elif self.reset_mode == 'none':
            return False
        elif self.reset_mode == 'periodic':
            # Reset every N batches (e.g., every 10 batches)
            return self.batch_count > 0 and self.batch_count % self.reset_frequency == 0
        elif self.reset_mode == 'confidence':
            if len(self.confidence_history) >= 5:  # Need history
                avg_confidence = sum(self.confidence_history[-5:]) / 5
                return avg_confidence < self.confidence_threshold
            return False
        elif self.reset_mode == 'hybrid':
            # Reset periodically OR when confidence drops
            periodic_reset = self.batch_count > 0 and self.batch_count % self.reset_frequency == 0
            confidence_reset = False
            if len(self.confidence_history) >= 5:
                avg_confidence = sum(self.confidence_history[-5:]) / 5
                confidence_reset = avg_confidence < self.confidence_threshold
            return periodic_reset or confidence_reset
        elif self.reset_mode == 'ema':
            # EMA mode doesn't use hard resets
            return False
        else:
            return False

    def _update_tracking(self, logits):
        """Update batch count, sample count, and confidence history."""
        self.batch_count += 1  # Increment batch counter
        self.sample_count += logits.shape[0]  # Track total samples for reference
        
        with torch.no_grad():
            probs = logits.softmax(dim=1)
            batch_confidence = probs.max(dim=1)[0].mean().item()
            self.confidence_history.append(batch_confidence)
            
            # Keep only recent history
            if len(self.confidence_history) > 50:
                self.confidence_history = self.confidence_history[-50:]

    def _ema_update(self):
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
    
    def compute_consensus_similarity(self, similarities):
        """Compute consensus-based similarity across sub-prototypes.
        
        Args:
            similarities: [Batch, Prototypes, Sub-prototypes] or [Batch, Prototypes]
        
        Returns:
            aggregated_sims: [Batch, Prototypes] - consensus similarity per prototype
        """
        if similarities.dim() == 2:
            # No sub-prototypes, return as-is
            return similarities
        
        # similarities: [B, P, K]
        if self.consensus_strategy == 'max':
            # Original: take best sub-prototype
            aggregated_sims, _ = similarities.max(dim=2)
        
        elif self.consensus_strategy == 'mean':
            # Consensus: all sub-prototypes must agree (average)
            aggregated_sims = similarities.mean(dim=2)
        
        elif self.consensus_strategy == 'median':
            # Robust consensus: median across sub-prototypes
            aggregated_sims = similarities.median(dim=2)[0]
        
        elif self.consensus_strategy == 'top_k_mean':
            # Soft consensus: average of top-K sub-prototypes
            K = similarities.shape[2]
            top_k = max(1, int(K * self.consensus_ratio))
            top_sims, _ = torch.topk(similarities, k=top_k, dim=2)
            aggregated_sims = top_sims.mean(dim=2)
        
        elif self.consensus_strategy == 'weighted_mean':
            # Weighted by similarity: high similarities contribute more
            # Use softmax as weights
            weights = F.softmax(similarities * 10, dim=2)  # Temperature=0.1
            aggregated_sims = (similarities * weights).sum(dim=2)
        
        else:
            raise ValueError(f"Unknown consensus_strategy: {self.consensus_strategy}")
        
        return aggregated_sims

    def reset(self):
        """Reset model parameters to pretrained state."""
        # Only reset model parameters, NOT optimizer state
        # Preserves Adam momentum/variance for effective episodic updates
        self.model.load_state_dict(self.model_state, strict=True)
        
        # Reset EMA state if in EMA mode
        if self.reset_mode == 'ema':
            self.ema_state = None
    
    def get_geo_filter_stats(self):
        """Get geometric filtering statistics."""
        stats = self.geo_filter_stats.copy()
        if stats['total_samples'] > 0:
            stats['filter_rate'] = stats['filtered_samples'] / stats['total_samples']
            if stats['min_similarities']:
                stats['overall_min_sim'] = min(stats['min_similarities'])
                stats['overall_max_sim'] = max(stats['max_similarities'])
                stats['overall_avg_sim'] = sum(stats['avg_similarities']) / len(stats['avg_similarities'])
            else:
                stats['overall_min_sim'] = None
                stats['overall_max_sim'] = None
                stats['overall_avg_sim'] = None
        else:
            stats['filter_rate'] = 0.0
            stats['overall_min_sim'] = None
            stats['overall_max_sim'] = None
            stats['overall_avg_sim'] = None
        return stats
    
    def reset_geo_filter_stats(self):
        """Reset geometric filtering statistics."""
        self.geo_filter_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'min_similarities': [],
            'max_similarities': [],
            'avg_similarities': []
        }
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward_no_adapt(self, x):
        """Forward pass without adaptation (used for metrics)."""
        outputs = self.model(x)
        if self.semantic_logit_blend <= 0:
            return outputs
        logits, min_distances, similarities = outputs
        logits = self._blend_semantic_logits(logits, similarities)
        return logits, min_distances, similarities

    def _prediction_activations(self, similarities):
        """Exactly reconstruct the activations consumed by ProtoViT's head."""
        if similarities.dim() == 2:
            return similarities
        slots = torch.sigmoid(self.model.patch_select * self.model.temp)
        num_subpatches = similarities.shape[-1]
        scale = slots * num_subpatches / slots.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        return (similarities * scale).sum(dim=-1)

    def _blend_semantic_logits(self, logits, similarities):
        activations = self._prediction_activations(similarities)
        quality = self.semantic_prototype_weights.to(activations.device)
        semantic_logits = F.linear(
            activations * quality.unsqueeze(0),
            self.model.last_layer.weight,
            self.model.last_layer.bias,
        )
        alpha = self.semantic_logit_blend
        if self.semantic_fusion == 'confidence_guard':
            # Preserve the native decision when suppressing low-quality
            # prototypes produces a weaker, conflicting decision.  Conversely,
            # retain VLM corrections when their class margin is at least as
            # strong.  This controller uses no target labels.
            with torch.no_grad():
                native_prob = logits.float().softmax(dim=1)
                semantic_prob = semantic_logits.float().softmax(dim=1)
                native_margin = native_prob.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
                semantic_margin = semantic_prob.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
                disagree = logits.argmax(dim=1).ne(semantic_logits.argmax(dim=1))
                accept = (~disagree) | (semantic_margin >= native_margin)
                alpha = logits.new_full((logits.shape[0], 1), alpha) * accept.unsqueeze(1)
        return (1.0 - alpha) * logits + alpha * semantic_logits


    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # 1. Forward Pass
        logits, min_distances, similarities = self.model(x)
        if self.semantic_logit_blend > 0:
            logits = self._blend_semantic_logits(logits, similarities)

        # 2. Identify Target Class (Pseudo-label)
        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1)

        # 3. Aggregate sub-prototypes using consensus strategy
        sim_scores = self.compute_consensus_similarity(similarities)

        # The compatibility student is distilled from label-blind VLM board
        # scores.  It is fixed and differentiation-free during test time.
        compatibility_weight = torch.ones(logits.shape[0], device=logits.device)
        if self.compatibility_student is not None:
            from semantic_prototype_guidance import runtime_features, predict_student
            compatibility_features = runtime_features(
                logits.detach(), sim_scores.detach(), proto_identities,
                self.model.last_layer.weight.detach(),
                self.semantic_prototype_weights.to(logits.device),
            )
            compatibility = predict_student(
                compatibility_features, self.compatibility_student
            )
            compatibility_weight = (
                self.compatibility_floor
                + (1.0 - self.compatibility_floor) * compatibility
            ).detach()

        # ========== Geometric Filtering: Filter unreliable samples ==========
        if self.use_geometric_filter:
            with torch.no_grad():
                # Use consensus similarity (already computed) for filtering
                # sim_scores is [B, P] after consensus aggregation
                # Find the best prototype match per sample
                max_sim_per_sample = sim_scores.max(dim=1)[0]  # [B]
                
                # Track statistics
                batch_size = max_sim_per_sample.shape[0]
                self.geo_filter_stats['total_samples'] += batch_size
                num_filtered = (max_sim_per_sample <= self.geo_filter_threshold).sum().item()
                self.geo_filter_stats['filtered_samples'] += num_filtered
                self.geo_filter_stats['min_similarities'].append(max_sim_per_sample.min().item())
                self.geo_filter_stats['max_similarities'].append(max_sim_per_sample.max().item())
                self.geo_filter_stats['avg_similarities'].append(max_sim_per_sample.mean().item())
                
                # Samples with low similarity to ALL prototypes are unreliable (noisy/corrupted)
                reliable_mask = (max_sim_per_sample > self.geo_filter_threshold).float()  # [B]
                
                # Track adapted samples and per-sample updates
                adapted_count = int(reliable_mask.sum().item())
                self.adaptation_stats['adapted_samples'] += adapted_count
                self.adaptation_stats['total_updates'] += adapted_count
                
                # If no reliable samples, skip adaptation
                if reliable_mask.sum() == 0:
                    return logits, min_distances, similarities
        else:
            # All samples are considered reliable
            reliable_mask = torch.ones(logits.shape[0], device=logits.device)
            # Track adapted samples and per-sample updates (all of them in this case)
            adapted_count = logits.shape[0]
            self.adaptation_stats['adapted_samples'] += adapted_count
            self.adaptation_stats['total_updates'] += adapted_count

        # 4. Masking: Target and Non-target prototypes
        if self.adapt_all_prototypes:
            # Adapt all prototypes (not just target prototypes)
            # Create mask of shape [B, P] with all ones
            batch_size = logits.shape[0]
            num_prototypes = proto_identities.shape[0]
            target_mask = torch.ones(batch_size, num_prototypes, device=logits.device, dtype=torch.float32)
        else:
            # Original behavior: only adapt target prototypes (matching predicted class)
            target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        nontarget_mask = 1.0 - target_mask
        
        # Apply sample reliability mask (broadcast to prototype dimension)
        sample_weights = reliable_mask.unsqueeze(1)  # [B, 1]

        # ========== PART A: Target Entropy Loss ==========
        eps = 1e-6
        
        # --- NEW: Ensemble Entropy (Voting across sub-prototypes) ---
        if self.use_ensemble_entropy and similarities.dim() == 3:
            # Treat each sub-prototype as a weak classifier
            # Compute entropy for each sub-prototype, then average
            # similarities: [B, P, K]
            
            # Apply target mask at sub-prototype level: [B, P, K]
            target_mask_3d = target_mask.unsqueeze(-1)  # [B, P, 1]
            masked_sims_3d = similarities * target_mask_3d  # [B, P, K]
            masked_sims_3d = torch.clamp(masked_sims_3d, min=-1.0, max=1.0)
            
            # Convert each sub-prototype similarity to probability
            proto_probs_3d = (masked_sims_3d + 1.0) / 2.0  # [B, P, K]
            proto_probs_3d = torch.clamp(proto_probs_3d, min=eps, max=1-eps)
            
            # Compute entropy for EACH sub-prototype independently
            entropy_per_subproto = -(proto_probs_3d * torch.log(proto_probs_3d) + 
                                     (1 - proto_probs_3d) * torch.log(1 - proto_probs_3d))  # [B, P, K]
            
            # Average entropies across sub-prototypes (ensemble)
            # This prevents a single overconfident sub-prototype from dominating
            entropy = entropy_per_subproto.mean(dim=2)  # [B, P]
            
        else:
            # Original: Aggregate first, then compute entropy
            masked_sims = sim_scores * target_mask
            masked_sims = torch.clamp(masked_sims, min=-1.0, max=1.0)
            proto_probs = (masked_sims + 1.0) / 2.0
            proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)
            
            # Minimize Binary Entropy for target prototypes
            entropy = -(proto_probs * torch.log(proto_probs) + 
                       (1 - proto_probs) * torch.log(1 - proto_probs))  # [B, P]
        
        # --- Prototype Importance Weighting ---
        if self.use_prototype_importance:
            # Get the last_layer weights: [num_classes, num_prototypes]
            # For each sample, get the weights for the predicted class
            last_layer_weights = self.model.last_layer.weight  # [num_classes, num_prototypes]
            
            # Get weights for predicted classes: [batch_size, num_prototypes]
            class_weights = last_layer_weights[pred_class]  # [B, P]
            
            # Normalize weights to [0, 1] range per sample (softmax-like but preserve relative importance)
            # Use absolute values since negative weights also indicate importance
            importance_weights = torch.abs(class_weights)

            # Downweight background/artifact prototypes using the one-time VLM
            # bank audit. The mask and renormalization preserve baseline scale.
            if self.use_semantic_prototype_weights:
                importance_weights = importance_weights * \
                    self.semantic_prototype_weights.to(importance_weights.device).unsqueeze(0)
            
            # Normalize to sum to 1 for each sample (only for target prototypes)
            importance_weights = importance_weights * target_mask
            importance_weights = importance_weights / (importance_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weight the entropy by prototype importance AND sample reliability
            weighted_entropy = entropy * target_mask * importance_weights * sample_weights
            loss_per_sample = weighted_entropy.sum(dim=1)  # [B]
        else:
            # Original: uniform weighting across all target prototypes, filtered by reliability
            masked_entropy = entropy * target_mask * sample_weights
            loss_per_sample = masked_entropy.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)  # [B]
        
        # --- Confidence Weighting ---
        if self.use_confidence_weighting:
            # Calculate prediction confidence (max probability)
            with torch.no_grad():
                probs = logits.softmax(dim=1)
                confidence = probs.max(dim=1)[0]  # [B]
            
            # Weight the loss by confidence AND reliability
            # Only count reliable samples in the mean
            proto_sample_loss = loss_per_sample * confidence * reliable_mask * compatibility_weight
            loss_target = proto_sample_loss.sum() / ((reliable_mask * compatibility_weight).sum() + 1e-8)
        else:
            # Only count reliable samples in the mean
            proto_sample_loss = loss_per_sample * reliable_mask * compatibility_weight
            loss_target = proto_sample_loss.sum() / ((reliable_mask * compatibility_weight).sum() + 1e-8)

        # Active semantic purification. Downweighting alone merely ignores a
        # spurious prototype; this term moves activation probability toward the
        # VLM-ranked meaningful prototypes and away from the bottom-ranked ones.
        loss_semantic = logits.new_tensor(0.0)
        if self.semantic_contrast_weight > 0:
            quality = self.semantic_prototype_weights.to(logits.device).unsqueeze(0)
            target_logits = (sim_scores / max(self.semantic_temperature, 1e-6)).masked_fill(
                ~target_mask.bool(), float('-inf')
            )
            activation_distribution = F.softmax(target_logits, dim=1)
            expected_quality = (activation_distribution * quality).sum(dim=1).clamp_min(1e-6)
            semantic_sample_loss = -torch.log(expected_quality)
            semantic_sample_weight = reliable_mask * compatibility_weight
            if self.use_confidence_weighting:
                semantic_sample_weight = semantic_sample_weight * confidence
            loss_semantic = (
                semantic_sample_loss * semantic_sample_weight
            ).sum() / semantic_sample_weight.sum().clamp_min(1e-8)

        # ========== PART B: Separation Loss (Push non-target prototypes to -1) ==========
        # For non-target prototypes, we want similarities to be close to -1 (dissimilar)
        # Map to [0, 1] and push toward 0
        nontarget_sims = sim_scores * nontarget_mask
        nontarget_sims = torch.clamp(nontarget_sims, min=-1.0, max=1.0)
        nontarget_probs = (nontarget_sims + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]
        nontarget_probs = torch.clamp(nontarget_probs, min=eps, max=1-eps)
        
        # Minimize the probability (push toward 0, meaning similarity toward -1)
        # Using binary cross-entropy with target=0, filtered by sample reliability
        separation_loss = -torch.log(1 - nontarget_probs) * nontarget_mask * sample_weights
        loss_separation = separation_loss.sum(dim=1) / (nontarget_mask.sum(dim=1) + 1e-8)
        loss_separation = (loss_separation * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)

        # ========== PART C: Sub-Prototype Coherence Loss ==========
        loss_coherence = 0.0
        if similarities.dim() == 3:
            # similarities: [Batch, Prototypes, Sub-prototypes]
            # For target prototypes, minimize variance across sub-prototypes
            target_similarities = similarities * target_mask.unsqueeze(-1)  # [B, P, K]
            
            # Calculate variance across sub-prototypes (dim=2) for each prototype
            # Only consider target prototypes
            mean_sub = target_similarities.sum(dim=2, keepdim=True) / (similarities.shape[2] + 1e-8)
            variance_sub = ((target_similarities - mean_sub) ** 2).sum(dim=2)  # [B, P]
            
            # Only count target prototypes, filtered by sample reliability
            coherence_loss = variance_sub * target_mask * sample_weights
            loss_coherence = coherence_loss.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)
            loss_coherence = (loss_coherence * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)

        # ========== PART D: Source Distribution KL Regularization ==========
        loss_source_kl = 0.0
        if self.alpha_source_kl > 0 and self.source_proto_stats is not None:
            # Compute KL divergence between test and source prototype activation distributions
            # sim_scores: [B, P] - current prototype similarities
            
            # Convert similarities to probabilities (softmax over prototypes)
            test_proto_probs = F.softmax(sim_scores / 0.1, dim=1)  # Temperature=0.1 for sharpness
            
            # Get source distribution (pre-computed on clean data)
            source_proto_probs = self.source_proto_stats['prototype_probs'].to(logits.device)  # [P]
            
            # KL(test || source): prevents test distribution from drifting too far from source
            # This acts like Fisher regularization but at the prototype activation level
            kl_div = F.kl_div(
                test_proto_probs.log(),  # Log of test distribution
                source_proto_probs.unsqueeze(0).expand(test_proto_probs.size(0), -1),  # Source distribution
                reduction='none'
            ).sum(dim=1)  # [B]
            
            # Only compute KL for reliable samples
            loss_source_kl = (kl_div * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
        
        # ========== Total Loss ==========
        # Balance the components using instance attributes
        loss = self.alpha_target * loss_target + self.alpha_separation * loss_separation
        if isinstance(loss_coherence, torch.Tensor):
            loss = loss + self.alpha_coherence * loss_coherence
        if isinstance(loss_source_kl, torch.Tensor) or (isinstance(loss_source_kl, (int, float)) and loss_source_kl != 0):
            print (f"Loss source KL: {loss_source_kl}")
            print (f"Loss: {loss}")
            loss = loss + self.alpha_source_kl * loss_source_kl

        proto_loss = loss + self.semantic_contrast_weight * loss_semantic
        logit_ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        logit_sample_weight = reliable_mask
        if self.shared_confidence_weighting and self.use_confidence_weighting:
            logit_sample_weight = logit_sample_weight * confidence
        output_sample_loss = logit_ent * logit_sample_weight
        loss_logit = output_sample_loss.sum() / (reliable_mask.sum() + 1e-8)

        class_target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        positive_scores = ((sim_scores.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp_min(1e-8)
        target_scores = positive_scores * class_target_mask
        q = target_scores / target_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
        proto_entropy = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=1)
        num_target = class_target_mask.sum(dim=1).clamp_min(2.0)
        concentration = 1.0 - proto_entropy / torch.log(num_target)
        strength = target_scores.max(dim=1).values / positive_scores.max(dim=1).values.clamp_min(1e-8)
        proto_reliability = (concentration * strength).clamp(0.0, 1.0)

        output_probs = logits.softmax(dim=1)
        output_reliability = 1.0 - (
            -(output_probs * torch.log(output_probs.clamp_min(1e-8))).sum(dim=1)
            / torch.log(logits.new_tensor(float(output_probs.shape[1])))
        )

        proto_weight = 1.0 - self.logit_weight
        logit_weight = self.logit_weight
        adaptive_lambda_per_sample = None
        proto_gradient_consistency = None
        output_gradient_consistency = None
        router_gate_fraction = None
        if self.adaptive_lambda:
            valid = reliable_mask.bool()
            if self.adaptive_lambda_strategy == 'activation_margin':
                masked_target = positive_scores.masked_fill(~class_target_mask.bool(), float('-inf'))
                k_target = min(
                    self.adaptive_topk,
                    int(class_target_mask.sum(dim=1).min().item()),
                )
                top_target = masked_target.topk(max(k_target, 1), dim=1).values
                delta = (top_target - 0.5).abs().mean(dim=1)
                adaptive_score = (delta / max(self.adaptive_delta0, 1e-8)).clamp(0.0, 1.0)
            elif self.adaptive_lambda_strategy == 'relative_reliability':
                adaptive_score = proto_reliability / (
                    proto_reliability + output_reliability + 1e-8
                )
            elif self.adaptive_lambda_strategy == 'relative_evidence':
                adaptive_score, proto_reliability, output_reliability = \
                    relative_evidence_lambda(
                        logits=logits,
                        similarities=positive_scores,
                        prototype_classes=proto_identities,
                        topk=self.adaptive_topk,
                    )
            elif self.adaptive_lambda_strategy == 'gradient_consistency':
                proto_gradient_consistency = self._gradient_consistency(
                    proto_sample_loss, reliable_mask
                )
                output_gradient_consistency = self._gradient_consistency(
                    output_sample_loss, reliable_mask
                )
                denominator = (
                    proto_gradient_consistency + output_gradient_consistency
                )
                if denominator.item() <= 1e-8:
                    gradient_lambda = self.lambda_ema if self.lambda_ema is not None else 0.5
                    gradient_lambda = logits.new_tensor(float(gradient_lambda))
                else:
                    gradient_lambda = proto_gradient_consistency / denominator
                adaptive_score = torch.full_like(
                    output_reliability, float(gradient_lambda.item())
                )
                # Reuse the common diagnostics as the actual controller signals.
                proto_reliability = torch.full_like(
                    proto_reliability, float(proto_gradient_consistency.item())
                )
                output_reliability = torch.full_like(
                    output_reliability, float(output_gradient_consistency.item())
                )
            elif self.adaptive_lambda_strategy in (
                    'source_free_router', 'source_free_router_absolute',
                    'source_free_router_coverage',
                    'source_free_router_coverage_absolute'):
                masked_target = positive_scores.masked_fill(
                    ~class_target_mask.bool(), float('-inf')
                )
                k_target = min(
                    self.adaptive_topk,
                    int(class_target_mask.sum(dim=1).min().item()),
                )
                top_target = masked_target.topk(max(k_target, 1), dim=1).values
                native_delta = (top_target - 0.5).abs().mean(dim=1)
                native_score = (
                    native_delta / max(self.adaptive_delta0, 1e-8)
                ).clamp(0.0, 1.0)
                relative_score, relative_proto_rel, relative_output_rel = \
                    relative_evidence_lambda(
                        logits=logits,
                        similarities=positive_scores,
                        prototype_classes=proto_identities,
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
                if self.adaptive_lambda_strategy in (
                        'source_free_router_absolute',
                        'source_free_router_coverage_absolute'):
                    # A favorable ratio is insufficient when both gradient
                    # signals are nearly incoherent.  The absolute floor is
                    # normalized and shared across every backbone.
                    gradient_gate = gradient_gate & (
                        proto_gradient_consistency
                        >= self.router_min_consistency
                    )
                if self.adaptive_lambda_strategy in (
                        'source_free_router_coverage',
                        'source_free_router_coverage_absolute'):
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
            if self.samplewise_lambda:
                proto_weight = current_lambda
                logit_weight = 1.0 - proto_weight
            else:
                if self.lambda_ema is None:
                    self.lambda_ema = current_lambda
                else:
                    self.lambda_ema = (self.lambda_ema_momentum * self.lambda_ema +
                                       (1.0 - self.lambda_ema_momentum) * current_lambda)
                proto_weight = min(self.lambda_max, max(self.lambda_min, self.lambda_ema))
                logit_weight = 1.0 - proto_weight

        need_grad_norms = self.gradient_normalize or self.record_diagnostics
        proto_grad_norm = self._gradient_norm(proto_loss) if need_grad_norms else None
        output_grad_norm = self._gradient_norm(loss_logit) if need_grad_norms else None
        proto_term = proto_loss
        output_term = loss_logit
        if self.gradient_normalize:
            proto_term = proto_loss / (proto_grad_norm.detach() + 1e-8)
            output_term = loss_logit / (output_grad_norm.detach() + 1e-8)

        if self.samplewise_lambda:
            if adaptive_lambda_per_sample is None:
                raise ValueError('samplewise_lambda requires adaptive_lambda=True')
            if self.alpha_separation != 0 or self.alpha_coherence != 0 or self.alpha_source_kl != 0:
                raise ValueError(
                    'samplewise_lambda currently supports the paper configuration '
                    'with target prototype entropy only'
                )
            proto_sample_term = self.alpha_target * proto_sample_loss
            output_sample_term = output_sample_loss
            if self.gradient_normalize:
                proto_sample_term = proto_sample_term / (proto_grad_norm.detach() + 1e-8)
                output_sample_term = output_sample_term / (output_grad_norm.detach() + 1e-8)
            loss = (
                adaptive_lambda_per_sample * proto_sample_term
                + (1.0 - adaptive_lambda_per_sample) * output_sample_term
            ).sum() / (reliable_mask.sum() + 1e-8)
        else:
            loss = proto_weight * proto_term + logit_weight * output_term

        if self.record_diagnostics or self.adaptive_lambda:
            valid = reliable_mask.bool()
            self.adaptation_stats['proto_loss'].append(float(proto_loss.detach().item()))
            self.adaptation_stats['output_loss'].append(float(loss_logit.detach().item()))
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
                    float(
                        router_gate_fraction
                        if router_gate_fraction is not None else
                        (proto_gradient_consistency.item() >= 2.0 * max(
                            output_gradient_consistency.item(), 1e-8
                        ))
                    )
                )

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, min_distances, similarities

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
        """Scale-invariant gradient agreement across two batch halves."""
        indices = reliable_mask.bool().nonzero(as_tuple=False).flatten()
        if indices.numel() < 4:
            return per_sample_loss.detach().new_tensor(0.0)
        first = indices[::2]
        second = indices[1::2]
        first_loss = per_sample_loss[first].mean()
        second_loss = per_sample_loss[second].mean()
        params = [
            p for group in self.optimizer.param_groups for p in group['params']
            if p.requires_grad
        ]
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


class ProtoEntropyEATA(nn.Module):
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
            outputs, min_distances, similarities = self.forward_and_adapt(x)

        return outputs, min_distances, similarities

    def reset(self):
        # Only reset model parameters, NOT optimizer state
        # Preserves Adam momentum/variance for effective episodic updates
        self.model.load_state_dict(self.model_state, strict=True)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # 1. Forward Pass
        logits, min_distances, similarities = self.model(x)

        # 2. Identify Target Class (Pseudo-label)
        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            
            # --- EATA-style thresholding ---
            num_classes = logits.shape[1]
            adaptive_threshold = self.entropy_threshold * torch.log(torch.tensor(num_classes, device=logits.device).float())
            
            probs = logits.softmax(dim=1)
            entropy_vals = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
            reliable_mask = (entropy_vals < adaptive_threshold).float()
            
            if reliable_mask.sum() == 0:
                return logits, min_distances, similarities
            
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1)

        # 3. Flatten/Max over sub-prototypes
        if similarities.dim() == 3:
            sim_scores, _ = similarities.max(dim=2) 
        else:
            sim_scores = similarities 

        # 4. Masks
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        nontarget_mask = 1.0 - target_mask
        sample_weights = reliable_mask.unsqueeze(1) 

        # --- PART A: Target Loss ---
        masked_sims = sim_scores * target_mask 
        masked_sims = torch.clamp(masked_sims, min=-1.0, max=1.0)
        proto_probs = (masked_sims + 1.0) / 2.0
        eps = 1e-6
        proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)

        entropy = -(proto_probs * torch.log(proto_probs) + 
                   (1 - proto_probs) * torch.log(1 - proto_probs))
        
        target_loss_map = entropy * target_mask * sample_weights
        
        # Normalize Target Loss by number of reliable SAMPLES
        denom_target = sample_weights.sum() + 1e-8
        loss_target = target_loss_map.sum() / denom_target

        # --- PART B: Negative Margin Loss ---
        neg_margin = 0.2 
        neg_violation = F.relu(sim_scores - neg_margin)
        
        # Apply masks
        neg_loss_map = neg_violation * nontarget_mask * sample_weights
        
        # Count how many negative prototypes contribute to the loss
        num_neg_prototypes = nontarget_mask.sum(dim=1).mean()
        denom_neg = (sample_weights.sum() * num_neg_prototypes) + 1e-8
        
        loss_neg = neg_loss_map.sum() / denom_neg
        
        # --- Total Loss ---
        loss = loss_target + loss_neg

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, min_distances, similarities
# Helper functions
def collect_params(model, adaptation_mode='layernorm_only'):
    """Collect parameters for test-time adaptation.
    
    Args:
        model: The ProtoViT model
        adaptation_mode: What to adapt during TTA
            'layernorm_only' - Only LayerNorm/BatchNorm (default, safest)
            'layernorm_proto' - LayerNorms + Prototype vectors
            'layernorm_proto_patch' - LayerNorms + Prototypes + Patch select
            'layernorm_proto_last' - LayerNorms + Prototypes + Last layer
            'layernorm_attn_bias' - LayerNorms + Attention biases in backbone
            'layernorm_last_block' - LayerNorms + Last transformer block
            'full_proto' - Prototypes + Patch select + Last layer (no backbone)
            'all_adaptive' - Everything except frozen backbone features
    
    Returns:
        params: List of parameters to optimize
        names: List of parameter names
    """
    params = []
    names = []

    # A literal full-model ablation for the paper.  Keep this distinct from
    # the legacy ``all_adaptive`` mode, which only enables prototypes, patch
    # selection, and the classifier head.
    if adaptation_mode == 'all_parameters':
        for name, parameter in model.named_parameters():
            params.append(parameter)
            names.append(name)
        return params, names
    
    # Always include LayerNorm/BatchNorm if mode includes 'layernorm'
    if 'layernorm' in adaptation_mode:
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
    
    # Add prototype vectors
    if 'proto' in adaptation_mode or adaptation_mode == 'all_adaptive':
        if hasattr(model, 'prototype_vectors'):
            params.append(model.prototype_vectors)
            names.append('prototype_vectors')
    
    # Add patch selection parameters
    if 'patch' in adaptation_mode or adaptation_mode == 'all_adaptive':
        if hasattr(model, 'patch_select'):
            params.append(model.patch_select)
            names.append('patch_select')
    
    # Add last layer (classification head)
    if 'last' in adaptation_mode or adaptation_mode == 'all_adaptive':
        if hasattr(model, 'last_layer'):
            for np, p in model.last_layer.named_parameters():
                params.append(p)
                names.append(f"last_layer.{np}")
    
    # Add attention biases from backbone
    if 'attn_bias' in adaptation_mode:
        for nm, m in model.named_modules():
            if 'attn' in nm.lower() or 'attention' in nm.lower():
                for np, p in m.named_parameters():
                    if 'bias' in np:  # Only biases, not full weights
                        params.append(p)
                        names.append(f"{nm}.{np}")
    
    # Add last transformer block (more aggressive)
    if 'last_block' in adaptation_mode:
        for nm, p in model.named_parameters():
            # Assuming blocks are named like 'features.blocks.11.*' (last block)
            if 'blocks.11' in nm or 'blocks.10' in nm:  # Last 2 blocks
                params.append(p)
                names.append(nm)
    
    return params, names


def configure_model(model, adaptation_mode='layernorm_only'):
    """Configure model for use with ProtoEntropy adaptation.
    
    Args:
        adaptation_mode: What parameters to enable for adaptation (same as collect_params)
    """
    # Gradients work in eval mode; this disables DropPath/dropout so repeated
    # online adaptation trajectories are deterministic.
    model.eval()
    # disable grad, to (re-)enable only what ProtoEntropy updates
    model.requires_grad_(False)

    if adaptation_mode == 'all_parameters':
        model.requires_grad_(True)
        return model
    
    # Configure LayerNorms/BatchNorms
    if 'layernorm' in adaptation_mode:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
    
    # Enable prototype vectors
    if 'proto' in adaptation_mode or adaptation_mode == 'all_adaptive':
        if hasattr(model, 'prototype_vectors'):
            model.prototype_vectors.requires_grad = True
    
    # Enable patch selection
    if 'patch' in adaptation_mode or adaptation_mode == 'all_adaptive':
        if hasattr(model, 'patch_select'):
            model.patch_select.requires_grad = True
    
    # Enable last layer
    if 'last' in adaptation_mode or adaptation_mode == 'all_adaptive':
        if hasattr(model, 'last_layer'):
            for p in model.last_layer.parameters():
                p.requires_grad = True
    
    # Enable attention biases
    if 'attn_bias' in adaptation_mode:
        for nm, m in model.named_modules():
            if 'attn' in nm.lower() or 'attention' in nm.lower():
                for np, p in m.named_parameters():
                    if 'bias' in np:
                        p.requires_grad = True
    
    # Enable last transformer blocks
    if 'last_block' in adaptation_mode:
        for nm, p in model.named_parameters():
            if 'blocks.11' in nm or 'blocks.10' in nm:
                p.requires_grad = True
    
    return model


def compute_source_proto_stats(model, source_loader, device, num_samples=500):
    """Compute prototype activation statistics on source/clean data.
    
    This is similar to EATA's Fisher computation but specifically for ProtoViT:
    - Collects prototype similarity distributions on clean data
    - Used as anchor to prevent test-time drift
    
    Args:
        model: The ProtoViT model
        source_loader: DataLoader with clean/source data
        device: Device to run on
        num_samples: Number of samples to use (default: 500)
    
    Returns:
        Dictionary with statistics: {
            'prototype_probs': Average probability distribution over prototypes,
            'similarity_mean': Mean similarity per prototype,
            'similarity_std': Std similarity per prototype
        }
    """
    model.eval()
    
    all_similarities = []
    total_samples = 0
    
    print(f"Computing source prototype statistics on {num_samples} clean samples...")
    
    with torch.no_grad():
        for images, _ in source_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            if total_samples >= num_samples:
                break
            
            # Forward pass to get prototype similarities
            logits, _, similarities = model(images)
            
            # Aggregate sub-prototypes (using max, consistent with default)
            if similarities.dim() == 3:
                sim_scores, _ = similarities.max(dim=2)  # [B, P]
            else:
                sim_scores = similarities
            
            all_similarities.append(sim_scores.cpu())
            total_samples += batch_size
    
    # Concatenate all similarities
    all_similarities = torch.cat(all_similarities, dim=0)[:num_samples]  # [N, P]
    
    # Compute statistics
    stats = {}
    
    # 1. Average probability distribution over prototypes (softmax)
    #    This represents the "typical" prototype activation pattern on clean data
    proto_probs_per_sample = F.softmax(all_similarities / 0.1, dim=1)  # [N, P]
    stats['prototype_probs'] = proto_probs_per_sample.mean(dim=0)  # [P]
    
    # 2. Mean and std of similarities per prototype
    stats['similarity_mean'] = all_similarities.mean(dim=0)  # [P]
    stats['similarity_std'] = all_similarities.std(dim=0)  # [P]
    
    # 3. Optional: Adaptive threshold based on source data
    #    Use mean - 2*std as a data-driven threshold
    max_sims_per_sample = all_similarities.max(dim=1)[0]  # [N]
    stats['adaptive_threshold'] = max_sims_per_sample.mean() - 2 * max_sims_per_sample.std()
    stats['mean_max_similarity'] = max_sims_per_sample.mean()
    stats['std_max_similarity'] = max_sims_per_sample.std()
    
    print(f"Source stats computed on {total_samples} samples:")
    print(f"  Mean max similarity: {stats['mean_max_similarity']:.4f}")
    print(f"  Std max similarity: {stats['std_max_similarity']:.4f}")
    print(f"  Suggested adaptive threshold: {stats['adaptive_threshold'].item():.4f}")
    
    return stats


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
