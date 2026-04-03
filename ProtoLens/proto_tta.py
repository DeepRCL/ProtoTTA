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


class ProtoTTA(nn.Module):
    """ProtoTTA adapts using prototype-aware binary entropy minimization."""
    
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 use_geometric_filter=True, geo_filter_threshold=0.3,
                 consensus_strategy='max', consensus_ratio=0.5,
                 importance_mode='global', sigmoid_temperature=5.0):
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
            }
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
                            sigmoid_temperature=5.0,
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
            return outputs.detach(), loss_mu, augmented_loss, similarities, 0
        except:
            # Total failure - return dummy outputs as last resort  
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
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            if adaptation_stats is not None:
                adaptation_stats['early_exit_stats']['nan_loss_count'] += 1
            optimizer.zero_grad()
            return outputs.detach(), loss_mu, augmented_loss, similarities, 0
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


def copy_model_and_optimizer(model, optimizer):
    """Copy model and optimizer states."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state
