"""
ProtoTTA (Prototype-Aware Test-Time Adaptation) for ProtoLens.
Adapted from ProtoViT for text classification.

V3 Configuration (Best from experiments):
- Geometric filtering: threshold=0.5 (adjusted for cosine similarity range [-1, 1])
- Consensus strategy: top_k_mean (top 50% of prototypes)
- Adaptation mode: LayerNorm + Attention biases
- Uses actual prototype similarities (NOT approximated from FC layer)

Key differences from ProtoViT:
1. ProtoLens uses cosine similarity in [-1, 1] range
2. ProtoLens returns (logits, loss_mu, augmented_loss, similarity)
3. ProtoLens has different forward signature (text inputs instead of images)
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def relative_evidence_lambda(logits, proto_probs, classifier_weights,
                             topk=3, eps=1e-8):
    """Relative evidence for shared ProtoLens prototypes.

    A prototype supports a class when that class has a positive classifier
    weight for it. This is the shared-prototype analogue of the explicit
    prototype-to-class assignments used by ProtoPNet/ProtoViT/ProtoPFormer.
    """
    num_classes = logits.shape[1]
    class_scores = []
    for class_index in range(num_classes):
        support = classifier_weights[class_index] > 0
        if support.any():
            per_class = proto_probs[:, support]
        else:
            # Keep the statistic defined for unusual checkpoints with no
            # positive weight by using the most class-associated prototypes.
            k_fallback = min(max(int(topk), 1), classifier_weights.shape[1])
            indices = classifier_weights[class_index].topk(k_fallback).indices
            per_class = proto_probs[:, indices]
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

    top2 = logits.softmax(dim=1).topk(2, dim=1).values
    output_reliability = (
        (top2[:, 0] - top2[:, 1]) / top2[:, 0].clamp_min(eps)
    ).clamp(0.0, 1.0)
    coefficient = proto_reliability / (
        proto_reliability + output_reliability + eps
    )
    return coefficient.detach(), proto_reliability, output_reliability


class ProtoTTA(nn.Module):
    """ProtoTTA adapts using prototype-aware binary entropy minimization."""
    
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 use_geometric_filter=True, geo_filter_threshold=0.3,
                 consensus_strategy='max', consensus_ratio=0.5,
                 importance_mode='global', sigmoid_temperature=5.0,
                 logit_weight=0.0, adaptive_lambda=False,
                 samplewise_lambda=False, gradient_normalize=False,
                 adaptive_delta0=0.25, adaptive_topk=3,
                 router_min_consistency=0.25,
                 lambda_ema_momentum=0.0, lambda_min=0.0,
                 lambda_max=1.0, record_diagnostics=False,
                 adaptive_lambda_strategy='activation_margin'):
        """
        Args:
            model: ProtoLens model (must return similarity as 4th output)
            optimizer: Optimizer for adaptation
            steps: Number of adaptation steps per batch
            episodic: If True, reset after each batch
            use_geometric_filter: Filter unreliable samples based on prototype similarity
            geo_filter_threshold: Minimum similarity threshold (in actual similarity range, not normalized)
                                 NOTE: ProtoLens uses non-normalized prototypes, so actual similarity range
                                 may vary. Typical values from tuning: 0.05-0.3 depending on dataset.
                                 Lower values = more selective filtering (fewer samples adapted)
                                 If None, uses adaptive threshold (25th percentile of consensus sims)
            consensus_strategy: How to aggregate prototype similarities into single score for filtering
                               'max': Use best prototype match (most selective, default)
                               'mean': Use average across all prototypes (less selective)
                               'top_k_mean': Use average of top k prototypes
                               NOTE: This is repurposed from ProtoViT's sub-prototype aggregation.
                                     In ProtoLens, it aggregates ACROSS prototypes, not sub-prototypes.
            consensus_ratio: Fraction of top prototypes (only used if consensus_strategy='top_k_mean')
            importance_mode: How to weight prototype importance (legacy parameter)
            sigmoid_temperature: Temperature for sigmoid on similarities. Higher = wider probability spread.
                                Default 5.0. Try 3-10 range. Since similarities are typically in [-0.4, 0.3],
                                temperature=5 maps this to approximately [0.12, 0.82] probability range.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        
        # Filtering configuration
        self.use_geometric_filter = use_geometric_filter
        self.geo_filter_threshold = geo_filter_threshold
        self.consensus_strategy = consensus_strategy
        self.consensus_ratio = consensus_ratio
        self.importance_mode = importance_mode
        
        # Sigmoid temperature for loss computation
        self.sigmoid_temperature = sigmoid_temperature

        # Unified λ: fraction of logit-entropy in total loss (0 = pure proto, 1 = pure logit)
        self.logit_weight = logit_weight
        self.adaptive_lambda = adaptive_lambda
        self.samplewise_lambda = samplewise_lambda
        self.gradient_normalize = gradient_normalize
        self.adaptive_delta0 = adaptive_delta0
        self.adaptive_topk = adaptive_topk
        self.router_min_consistency = router_min_consistency
        self.lambda_ema_momentum = lambda_ema_momentum
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.record_diagnostics = record_diagnostics
        self.adaptive_lambda_strategy = adaptive_lambda_strategy
        
        # Filter mode - 'geometric' or 'none'
        self.filter_mode = 'geometric' if use_geometric_filter else 'none'
        
        # Save initial state
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Adaptation tracking
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,  # Samples passing filter
            'total_updates': 0,
            'filter_stats': {
                'filtered_out': 0,
                'avg_similarity': [],
                'avg_confidence': []
            },
            'proto_loss': [],
            'output_loss': [],
            'proto_grad_norm': [],
            'output_grad_norm': [],
            'adaptive_lambda': [],
            'adaptive_lambda_raw': [],
            'adaptive_margin': [],
            'adaptive_saturation_rate': [],
            'proto_signal_reliability': [],
            'output_signal_reliability': [],
            'proto_gradient_consistency': [],
            'output_gradient_consistency': [],
            'adaptive_router_gate': [],
        }

    def forward(self, input_ids=None, attention_mask=None, special_tokens_mask=None,
                mode="test", original_text=None, current_batch_num=None, **kwargs):
        """Forward pass with ProtoTTA adaptation for text inputs."""
        if self.episodic:
            self.reset()
        
        batch_size = input_ids.size(0) if input_ids is not None else 1
        self.adaptation_stats['total_samples'] += batch_size
        
        for _ in range(self.steps):
            outputs, proto_dist, proto_val, similarity, num_adapted = forward_and_adapt_proto(
                model=self.model,
                optimizer=self.optimizer,
                use_geometric_filter=self.use_geometric_filter,
                geo_filter_threshold=self.geo_filter_threshold,
                consensus_strategy=self.consensus_strategy,
                consensus_ratio=self.consensus_ratio,
                importance_mode=self.importance_mode,
                sigmoid_temperature=self.sigmoid_temperature,
                logit_weight=self.logit_weight,
                adaptive_lambda=self.adaptive_lambda,
                samplewise_lambda=self.samplewise_lambda,
                gradient_normalize=self.gradient_normalize,
                adaptive_delta0=self.adaptive_delta0,
                adaptive_topk=self.adaptive_topk,
                router_min_consistency=self.router_min_consistency,
                lambda_ema_momentum=self.lambda_ema_momentum,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
                record_diagnostics=self.record_diagnostics,
                adaptive_lambda_strategy=self.adaptive_lambda_strategy,
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode=mode,
                original_text=original_text,
                current_batch_num=current_batch_num,
                adaptation_stats=self.adaptation_stats,
                **kwargs
            )
            self.adaptation_stats['adapted_samples'] += num_adapted
            if num_adapted > 0:
                self.adaptation_stats['total_updates'] += 1

        return outputs, proto_dist, proto_val, similarity

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.adaptation_stats.pop('_lambda_ema', None)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def compute_consensus_similarity(similarities, strategy='top_k_mean', ratio=0.5):
    """Aggregate prototype similarities into a single reliability score per sample.
    
    NOTE: This function is repurposed from ProtoViT (where it aggregated sub-prototypes).
    In ProtoLens, it aggregates ACROSS prototypes to get a single score for geometric filtering.
    
    Args:
        similarities: [Batch, Prototypes] tensor (cosine similarity in [-1, 1])
        strategy: How to aggregate prototypes into single score:
                 'max': Use best prototype match (most selective)
                 'mean': Use average across all prototypes (less selective)
                 'top_k_mean': Use average of top k prototypes
        ratio: Fraction of top prototypes to use (only for 'top_k_mean')
    
    Returns:
        aggregated_sims: [Batch] - single reliability score per sample for filtering
    """
    if similarities.dim() == 1:
        return similarities
    elif similarities.dim() == 2:
        # [Batch, Prototypes] - aggregate across prototypes
        if strategy == 'top_k_mean':
            # Top-k mean: average of top ratio% prototypes
            k = max(1, int(similarities.size(1) * ratio))
            topk_values, _ = torch.topk(similarities, k, dim=1)
            return topk_values.mean(dim=1)  # [Batch]
        elif strategy == 'max':
            return similarities.max(dim=1)[0]  # [Batch]
        elif strategy == 'mean':
            return similarities.mean(dim=1)  # [Batch]
        else:
            return similarities.max(dim=1)[0]  # Default: max
    else:
        raise ValueError(f"Unexpected similarity shape: {similarities.shape}")


def binary_entropy_loss(similarities, epsilon=1e-8):
    """Compute binary entropy loss from prototype similarities.
    
    NOTE: For cosine similarity in [-1, 1], we first normalize to [0, 1].
    Encourages decisive activations (close to 0 or 1 after normalization).
    
    Args:
        similarities: [Batch, Prototypes] - cosine similarities in [-1, 1]
    
    Returns:
        loss: Scalar entropy loss
    """
    # Normalize from [-1, 1] to [0, 1]
    p = (similarities + 1.0) / 2.0
    # Clip to avoid log(0)
    p = torch.clamp(p, epsilon, 1.0 - epsilon)
    
    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    
    return entropy.mean()


@torch.enable_grad()
def forward_and_adapt_proto(model, optimizer, use_geometric_filter, geo_filter_threshold,
                            consensus_strategy, consensus_ratio, importance_mode='global',
                            sigmoid_temperature=5.0, logit_weight=0.0,
                            adaptive_lambda=False, samplewise_lambda=False,
                            gradient_normalize=False, adaptive_delta0=0.25,
                            adaptive_topk=3, router_min_consistency=0.25,
                            lambda_ema_momentum=0.0,
                            lambda_min=0.0, lambda_max=1.0,
                            record_diagnostics=False,
                            adaptive_lambda_strategy='activation_margin',
                            input_ids=None, attention_mask=None, special_tokens_mask=None,
                            mode="test", original_text=None, current_batch_num=None,
                            adaptation_stats=None, **kwargs):
    """Forward and adapt model using prototype-aware loss.
    
    This is the CORRECTED version that uses actual prototype similarities
    returned by ProtoLens, matching the ProtoViT approach.
    
    Returns:
        outputs: Model predictions
        proto_dist: Prototype distances (loss_mu from ProtoLens)
        proto_val: Prototype values (augmented_loss from ProtoLens)
        similarity: Prototype similarities (for metrics)
        num_adapted: Number of samples that were adapted
    """
    # Initialize early exit tracking if not present
    if adaptation_stats is not None:
        if 'early_exit_stats' not in adaptation_stats:
            adaptation_stats['early_exit_stats'] = {
                'exception_count': 0,
                'nan_outputs_count': 0,
                'nan_similarities_count': 0,
                'nan_loss_count': 0,
                'nan_grad_count': 0
            }
    
    # Forward pass - ProtoLens returns (logits, loss_mu, augmented_loss, similarity)
    try:
        outputs, loss_mu, augmented_loss, similarities = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            mode=mode,
            original_text=original_text,
            current_batch_num=current_batch_num,
            **kwargs
        )
    except (ValueError, RuntimeError) as e:
        # If forward pass fails, try running without gradients to get valid outputs
        if adaptation_stats is not None:
            adaptation_stats['early_exit_stats']['exception_count'] += 1
        print(f"[PROTOTTA EXCEPTION] forward() with grad raised {type(e).__name__}: {e}")
        optimizer.zero_grad()
        
        # Try to get valid outputs without gradients
        try:
            with torch.no_grad():
                outputs, loss_mu, augmented_loss, similarities = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    mode=mode,
                    original_text=original_text,
                    current_batch_num=current_batch_num,
                    **kwargs
                )
            print(f"[PROTOTTA EXCEPTION] retry without grad SUCCEEDED (num_adapted=0 this batch)")
            return outputs.detach(), loss_mu, augmented_loss, similarities, 0
        except Exception as e2:
            # Total failure - return dummy outputs as last resort
            print(f"[PROTOTTA EXCEPTION] retry without grad ALSO FAILED with {type(e2).__name__}: {e2} -> returning DUMMY ZERO outputs")
            batch_size = input_ids.size(0) if input_ids is not None else 1
            num_classes = 2  # Binary classification
            device = input_ids.device if input_ids is not None else 'cpu'
            # Get actual number of prototypes from model
            num_prototypes = model.num_prototypes if hasattr(model, 'num_prototypes') else 50
            dummy_outputs = torch.zeros(batch_size, num_classes, device=device)
            dummy_similarities = torch.zeros(batch_size, num_prototypes, device=device)
            return dummy_outputs, torch.tensor(0.0), torch.tensor(0.0), dummy_similarities, 0
    
    batch_size = outputs.size(0)
    
    # Check for NaN/Inf in outputs
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        if adaptation_stats is not None:
            adaptation_stats['early_exit_stats']['nan_outputs_count'] += 1
        optimizer.zero_grad()
        return outputs.detach(), loss_mu, augmented_loss, similarities, 0
    
    # Check for NaN/Inf in similarities
    if torch.isnan(similarities).any() or torch.isinf(similarities).any():
        if adaptation_stats is not None:
            adaptation_stats['early_exit_stats']['nan_similarities_count'] += 1
        optimizer.zero_grad()
        return outputs.detach(), loss_mu, augmented_loss, similarities, 0
    
    # similarities is [Batch, Prototypes] with cosine similarity in [-1, 1]
    # Aggregate across prototypes to get single reliability score per sample
    # (This is repurposed from ProtoViT's sub-prototype aggregation, but here it's
    #  used to aggregate across prototypes for geometric filtering)
    consensus_sims = compute_consensus_similarity(
        similarities, 
        strategy=consensus_strategy,
        ratio=consensus_ratio
    )  # [Batch] - one score per sample
    
    # ========== Geometric Filtering ==========
    if use_geometric_filter:
        # Track statistics
        if adaptation_stats is not None:
            adaptation_stats['filter_stats']['avg_similarity'].append(consensus_sims.mean().item())
            # Track min/max for debugging
            if 'consensus_min' not in adaptation_stats['filter_stats']:
                adaptation_stats['filter_stats']['consensus_min'] = []
                adaptation_stats['filter_stats']['consensus_max'] = []
            adaptation_stats['filter_stats']['consensus_min'].append(consensus_sims.min().item())
            adaptation_stats['filter_stats']['consensus_max'].append(consensus_sims.max().item())
        
        # Adaptive threshold: if threshold is None, use percentile-based threshold
        if geo_filter_threshold is None:
            # Use 25th percentile as threshold (adaptive to batch distribution)
            threshold = torch.quantile(consensus_sims, 0.25).item()
            if adaptation_stats is not None:
                if 'adaptive_threshold' not in adaptation_stats['filter_stats']:
                    adaptation_stats['filter_stats']['adaptive_threshold'] = []
                adaptation_stats['filter_stats']['adaptive_threshold'].append(threshold)
        else:
            threshold = geo_filter_threshold
        
        # Filter: keep samples with high prototype similarity
        # threshold is in actual similarity range (may not be [-1, 1])
        reliable_mask = consensus_sims >= threshold  # [Batch]
        num_reliable = reliable_mask.sum().item()
        
        if adaptation_stats is not None:
            adaptation_stats['filter_stats']['filtered_out'] += (batch_size - num_reliable)
            # Track how many samples pass the threshold
            if 'samples_above_threshold' not in adaptation_stats['filter_stats']:
                adaptation_stats['filter_stats']['samples_above_threshold'] = []
            adaptation_stats['filter_stats']['samples_above_threshold'].append(num_reliable)
        
        if num_reliable == 0:
            # No reliable samples, skip adaptation
            # Debug: print first batch to understand why
            if adaptation_stats is not None and adaptation_stats.get('total_samples', 0) <= batch_size:
                print(f"\n[WARNING] Geometric filter filtered out all {batch_size} samples in batch!")
                print(f"  Consensus sims range: [{consensus_sims.min().item():.4f}, {consensus_sims.max().item():.4f}]")
                print(f"  Consensus sims mean: {consensus_sims.mean().item():.4f}")
                print(f"  Threshold: {geo_filter_threshold}")
                print(f"  Strategy: {consensus_strategy}")
                print(f"  Similarities range: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
                print(f"  Suggestion: Try lowering threshold to ~{consensus_sims.max().item() * 0.8:.4f} or disable geometric filter")
            optimizer.zero_grad()
            return outputs.detach(), loss_mu, augmented_loss, similarities, 0
        
        # Use only reliable samples for adaptation
        similarities_filtered = similarities[reliable_mask]
        outputs_filtered = outputs[reliable_mask]
    else:
        similarities_filtered = similarities
        outputs_filtered = outputs
        num_reliable = batch_size
    
    # ========== Compute ProtoTTA Loss (Class-Aware) ==========
    # Key insight: Unlike ProtoViT which has explicit prototype-class assignments,
    # ProtoLens uses FC layer weights to determine how prototypes contribute to each class.
    # 
    # For the PREDICTED class:
    #   - Prototypes with POSITIVE FC weights should have HIGH similarity (support the prediction)
    #   - Prototypes with NEGATIVE FC weights should have LOW similarity (don't contradict)
    #
    # This is a DIRECTED loss that aligns with the model's decision-making.
    
    if num_reliable > 0:
        eps = 1e-6
        sims_clamped = torch.clamp(similarities_filtered, min=-1.0, max=1.0)
        
        # Track statistics for debugging
        if adaptation_stats is not None:
            if 'similarity_stats' not in adaptation_stats:
                adaptation_stats['similarity_stats'] = {
                    'min': [],
                    'max': [],
                    'mean': [],
                    'std': []
                }
            adaptation_stats['similarity_stats']['min'].append(sims_clamped.min().item())
            adaptation_stats['similarity_stats']['max'].append(sims_clamped.max().item())
            adaptation_stats['similarity_stats']['mean'].append(sims_clamped.mean().item())
            adaptation_stats['similarity_stats']['std'].append(sims_clamped.std().item())
        # =====================================================================
        # PROTOTYPE-BASED ENTROPY with SIGMOID and FC-WEIGHT TARGETS
        # =====================================================================
        # Key insight: Use temperature-scaled sigmoid to spread out probabilities
        # from the narrow similarity range, and use FC weights to determine
        # which direction to push each prototype.
        #
        # For predicted class:
        #   - Prototypes with POSITIVE FC weight: push similarity UP (toward 1)
        #   - Prototypes with NEGATIVE FC weight: push similarity DOWN (toward 0)
        # =====================================================================
        
        # Temperature scaling: spreads out the narrow similarity range
        # Higher temperature = stronger push toward 0/1
        
        # Apply sigmoid with temperature to get probabilities
        # This maps similarities to [0, 1] with proper spread
        proto_probs = torch.sigmoid(sims_clamped * sigmoid_temperature)
        proto_probs = torch.clamp(proto_probs, min=eps, max=1.0 - eps)
        
        if hasattr(model, 'fc') and hasattr(model.fc, 'weight'):
            # Get FC weights and predicted classes
            fc_weights = model.fc.weight  # [num_classes, num_prototypes]
            pred_class = outputs_filtered.argmax(dim=1)  # [num_reliable]
            
            # Get FC weights for each sample's predicted class
            fc_weights_for_pred = fc_weights[pred_class]  # [num_reliable, num_prototypes]
            
            # Create targets based on FC weight signs
            # Positive weight → target = 1 (prototype supports prediction, want high similarity)
            # Negative weight → target = 0 (prototype opposes prediction, want low similarity)
            # Use sigmoid on weights to get soft targets (smoother gradients)
            targets = torch.sigmoid(fc_weights_for_pred * 2.0)  # Soft targets in [0, 1]
            targets = torch.clamp(targets, min=eps, max=1.0 - eps)
            
            # Importance weighting based on absolute FC weight magnitude
            # Prototypes with larger weights (either direction) are more important
            importance = torch.abs(fc_weights_for_pred)
            importance = importance / (importance.max(dim=1, keepdim=True)[0] + eps)
            
            # BCE loss: pushes proto_probs toward targets
            # For supporting prototypes (target~1): minimize -log(proto_probs) → increase probs
            # For opposing prototypes (target~0): minimize -log(1-proto_probs) → decrease probs
            bce_loss = -(targets * torch.log(proto_probs) + 
                        (1 - targets) * torch.log(1 - proto_probs))
            
            # Weight by importance
            weighted_loss = bce_loss * importance
            
            # Average over prototypes and samples
            loss = weighted_loss.mean()
            proto_loss_per_sample = weighted_loss.mean(dim=1)
            
            # Track statistics
            if adaptation_stats is not None:
                if 'loss_stats' not in adaptation_stats:
                    adaptation_stats['loss_stats'] = {
                        'proto_probs_mean': [],
                        'targets_mean': [],
                        'loss_value': []
                    }
                adaptation_stats['loss_stats']['proto_probs_mean'].append(proto_probs.mean().item())
                adaptation_stats['loss_stats']['targets_mean'].append(targets.mean().item())
                adaptation_stats['loss_stats']['loss_value'].append(loss.item())
        else:
            # Fallback: Simple binary entropy minimization (no FC layer)
            entropy = -(proto_probs * torch.log(proto_probs) + 
                       (1 - proto_probs) * torch.log(1 - proto_probs))
            loss = entropy.mean()
            proto_loss_per_sample = entropy.mean(dim=1)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            if adaptation_stats is not None:
                adaptation_stats['early_exit_stats']['nan_loss_count'] += 1
            optimizer.zero_grad()
            return outputs.detach(), loss_mu, augmented_loss, similarities, 0

        proto_loss = loss
        need_output_loss = logit_weight > 0 or adaptive_lambda
        loss_logit = None
        output_loss_per_sample = None
        if need_output_loss:
            logit_ent = -(outputs_filtered.softmax(1) * outputs_filtered.log_softmax(1)).sum(1)
            output_loss_per_sample = logit_ent
            loss_logit = logit_ent.mean()
            if torch.isnan(loss_logit) or torch.isinf(loss_logit):
                optimizer.zero_grad()
                return outputs.detach(), loss_mu, augmented_loss, similarities, 0

        need_grad_norms = gradient_normalize or record_diagnostics
        proto_grad_norm = _gradient_norm(proto_loss, optimizer) if need_grad_norms else None
        output_grad_norm = (
            _gradient_norm(loss_logit, optimizer)
            if need_grad_norms and loss_logit is not None else None
        )

        if adaptive_lambda:
            controller_proto_reliability = None
            controller_output_reliability = None
            router_gate_fraction = None
            router_strategies = (
                'source_free_router',
                'source_free_router_absolute',
                'source_free_router_evidence',
                'source_free_router_coverage',
                'source_free_router_coverage_absolute',
            )
            if adaptive_lambda_strategy == 'activation_margin' or \
                    adaptive_lambda_strategy in router_strategies:
                # Preserve the original activation-margin controller exactly.
                if hasattr(model, 'fc') and hasattr(model.fc, 'weight'):
                    support_mask = fc_weights_for_pred > 0
                    target_activations = proto_probs.masked_fill(
                        ~support_mask, float('-inf')
                    )
                    k = min(max(int(adaptive_topk), 1), target_activations.shape[1])
                    top_target = target_activations.topk(k, dim=1).values
                    valid_target = torch.isfinite(top_target)
                    target_margin = torch.where(
                        valid_target, (top_target - 0.5).abs(),
                        torch.zeros_like(top_target)
                    )
                    adaptive_margin = (
                        target_margin.sum(dim=1)
                        / valid_target.sum(dim=1).clamp_min(1)
                    )
                    missing_target = ~valid_target.any(dim=1)
                    if missing_target.any():
                        fallback = proto_probs[missing_target].topk(k, dim=1).values
                        adaptive_margin[missing_target] = (
                            fallback - 0.5
                        ).abs().mean(dim=1)
                else:
                    k = min(max(int(adaptive_topk), 1), proto_probs.shape[1])
                    top_target = proto_probs.topk(k, dim=1).values
                    adaptive_margin = (top_target - 0.5).abs().mean(dim=1)
                lambda_per_sample = (
                    adaptive_margin / max(adaptive_delta0, 1e-8)
                ).clamp(lambda_min, lambda_max).detach()
            if adaptive_lambda_strategy == 'relative_evidence':
                if not (hasattr(model, 'fc') and hasattr(model.fc, 'weight')):
                    raise ValueError('relative_evidence requires model.fc.weight')
                (lambda_per_sample,
                 controller_proto_reliability,
                 controller_output_reliability) = relative_evidence_lambda(
                    logits=outputs_filtered,
                    proto_probs=proto_probs,
                    classifier_weights=model.fc.weight,
                    topk=adaptive_topk,
                )
                lambda_per_sample = lambda_per_sample.clamp(
                    lambda_min, lambda_max
                )
                adaptive_margin = controller_proto_reliability
            elif adaptive_lambda_strategy in router_strategies:
                if not (hasattr(model, 'fc') and hasattr(model.fc, 'weight')):
                    raise ValueError(
                        f'{adaptive_lambda_strategy} requires model.fc.weight'
                    )
                native_lambda = lambda_per_sample
                native_margin = adaptive_margin
                (relative_lambda,
                 relative_proto_reliability,
                 relative_output_reliability) = relative_evidence_lambda(
                    logits=outputs_filtered,
                    proto_probs=proto_probs,
                    classifier_weights=model.fc.weight,
                    topk=adaptive_topk,
                )
                relative_lambda = relative_lambda.clamp(lambda_min, lambda_max)
                proto_gradient_consistency = _gradient_consistency(
                    proto_loss_per_sample, optimizer
                )
                output_gradient_consistency = _gradient_consistency(
                    output_loss_per_sample, optimizer
                )
                gradient_gate = (
                    proto_gradient_consistency
                    >= 2.0 * output_gradient_consistency.clamp_min(1e-8)
                )
                if adaptive_lambda_strategy in (
                        'source_free_router_absolute',
                        'source_free_router_coverage_absolute'):
                    gradient_gate = gradient_gate & (
                        proto_gradient_consistency >= router_min_consistency
                    )

                if adaptive_lambda_strategy == 'source_free_router_evidence':
                    # A saturated softmax margin is not evidence that output
                    # entropy is trustworthy.  Override it samplewise when the
                    # prototype space still separates the predicted class.
                    guard = (
                        (relative_output_reliability >= 0.90)
                        & (relative_proto_reliability >= 0.40)
                    )
                    native_gate = guard | bool(gradient_gate.item())
                elif adaptive_lambda_strategy in (
                        'source_free_router_coverage',
                        'source_free_router_coverage_absolute'):
                    # If only a small fraction passes the independent
                    # geometric filter, a saturated output margin is treated
                    # as overconfidence rather than reliability.
                    reliable_fraction = float(num_reliable) / max(batch_size, 1)
                    coverage_guard = (
                        relative_output_reliability.mean() >= 0.90
                    ) & (reliable_fraction <= 0.40)
                    native_gate = torch.full_like(
                        native_lambda, dtype=torch.bool,
                        fill_value=(
                            bool(gradient_gate.item())
                            or bool(coverage_guard.item())
                        ),
                    )
                else:
                    native_gate = torch.full_like(
                        native_lambda, dtype=torch.bool,
                        fill_value=bool(gradient_gate.item()),
                    )

                lambda_per_sample = torch.where(
                    native_gate, native_lambda, relative_lambda
                )
                adaptive_margin = torch.where(
                    native_gate, native_margin, relative_proto_reliability
                )
                controller_proto_reliability = relative_proto_reliability
                controller_output_reliability = relative_output_reliability
                router_gate_fraction = native_gate.float().mean().item()
            else:
                if adaptive_lambda_strategy != 'activation_margin':
                    raise ValueError(
                        f'Unknown adaptive lambda strategy: {adaptive_lambda_strategy}'
                    )

            current_lambda = lambda_per_sample.mean().item()

            if samplewise_lambda:
                proto_weight = current_lambda  # diagnostic mean only
                proto_sample_term = proto_loss_per_sample
                output_sample_term = output_loss_per_sample
                if gradient_normalize:
                    proto_sample_term = proto_sample_term / (proto_grad_norm.detach() + 1e-8)
                    output_sample_term = output_sample_term / (output_grad_norm.detach() + 1e-8)
                loss = (
                    lambda_per_sample * proto_sample_term
                    + (1.0 - lambda_per_sample) * output_sample_term
                ).mean()
            else:
                previous_lambda = (
                    adaptation_stats.get('_lambda_ema')
                    if adaptation_stats is not None else None
                )
                if previous_lambda is None:
                    smoothed_lambda = current_lambda
                else:
                    smoothed_lambda = (
                        lambda_ema_momentum * previous_lambda
                        + (1.0 - lambda_ema_momentum) * current_lambda
                    )
                proto_weight = min(lambda_max, max(lambda_min, smoothed_lambda))
                if adaptation_stats is not None:
                    adaptation_stats['_lambda_ema'] = proto_weight
                proto_term = proto_loss
                output_term = loss_logit
                if gradient_normalize:
                    proto_term = proto_term / (proto_grad_norm.detach() + 1e-8)
                    output_term = output_term / (output_grad_norm.detach() + 1e-8)
                loss = proto_weight * proto_term + (1.0 - proto_weight) * output_term

            if adaptation_stats is not None:
                adaptation_stats['adaptive_lambda'].append(float(proto_weight))
                adaptation_stats['adaptive_lambda_raw'].append(float(current_lambda))
                adaptation_stats['adaptive_margin'].append(float(adaptive_margin.mean().item()))
                saturation = ((lambda_per_sample <= lambda_min + 1e-8) |
                              (lambda_per_sample >= lambda_max - 1e-8)).float().mean()
                adaptation_stats['adaptive_saturation_rate'].append(float(saturation.item()))
                if controller_proto_reliability is not None:
                    adaptation_stats['proto_signal_reliability'].append(
                        float(controller_proto_reliability.mean().item())
                    )
                    adaptation_stats['output_signal_reliability'].append(
                        float(controller_output_reliability.mean().item())
                    )
                if adaptive_lambda_strategy in router_strategies:
                    adaptation_stats['proto_gradient_consistency'].append(
                        float(proto_gradient_consistency.item())
                    )
                    adaptation_stats['output_gradient_consistency'].append(
                        float(output_gradient_consistency.item())
                    )
                    adaptation_stats['adaptive_router_gate'].append(
                        float(
                            router_gate_fraction
                            if router_gate_fraction is not None else
                            (proto_gradient_consistency.item() >= 2.0 * max(
                                output_gradient_consistency.item(), 1e-8
                            ))
                        )
                    )
        elif logit_weight > 0:
            # Preserve the existing fixed-λ implementation exactly.
            loss = (1.0 - logit_weight) * proto_loss + logit_weight * loss_logit

        if adaptation_stats is not None and (record_diagnostics or adaptive_lambda):
            adaptation_stats['proto_loss'].append(float(proto_loss.detach().item()))
            adaptation_stats['output_loss'].append(
                float(loss_logit.detach().item()) if loss_logit is not None else 0.0
            )
            if proto_grad_norm is not None:
                adaptation_stats['proto_grad_norm'].append(float(proto_grad_norm.item()))
            if output_grad_norm is not None:
                adaptation_stats['output_grad_norm'].append(float(output_grad_norm.item()))
    else:
        loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)

    # Backward and update
    if num_reliable > 0 and loss.requires_grad:
        loss.backward()
        
        # Gradient clipping for stability
        params_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if len(params_with_grad) > 0:
            torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)
            
            # Check for NaN/Inf gradients
            has_nan_grad = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                              for p in params_with_grad if p.grad is not None)
            if not has_nan_grad:
                optimizer.step()
            else:
                if adaptation_stats is not None:
                    adaptation_stats['early_exit_stats']['nan_grad_count'] += 1
                optimizer.zero_grad()
                return outputs.detach(), loss_mu, augmented_loss, similarities, 0
        
        optimizer.zero_grad()
    else:
        optimizer.zero_grad()
    
    return outputs.detach(), loss_mu, augmented_loss, similarities, num_reliable


def _gradient_norm(loss, optimizer):
    """L2 norm of a loss gradient over exactly the adapted parameters."""
    params = [
        p for group in optimizer.param_groups for p in group['params']
        if p.requires_grad
    ]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    squared = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    if not squared:
        return loss.detach().new_tensor(0.0)
    return torch.stack(squared).sum().sqrt()


def _gradient_consistency(per_sample_loss, optimizer):
    """Scale-invariant gradient agreement across two reliable batch halves."""
    if per_sample_loss is None or per_sample_loss.numel() < 4:
        return per_sample_loss.detach().new_tensor(0.0)
    first_loss = per_sample_loss[::2].mean()
    second_loss = per_sample_loss[1::2].mean()
    params = [
        p for group in optimizer.param_groups for p in group['params']
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


def copy_model_and_optimizer(model, optimizer):
    """Copy model and optimizer states."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state
