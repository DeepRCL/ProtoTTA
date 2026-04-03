"""
EATA (Efficient Anti-Catastrophic Test-Time Adaptation) for ProtoLens.
Adapted from EATA ICML 2022 for text classification.

EATA filters samples based on:
1. Entropy threshold (reliability)
2. Cosine similarity to previous samples (redundancy)
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
import torch.nn.functional as F


class EATA(nn.Module):
    """EATA adapts a model by entropy minimization with sample filtering."""
    
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, 
                 steps=1, episodic=False, e_margin=None, d_margin=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Default e_margin for binary classification
        # Original EATA uses: log(num_classes)/2 - 1 for ImageNet (1000 classes)
        # For binary: log(2) ≈ 0.69, so log(2)/2 ≈ 0.35
        # But we want to keep confident samples (low entropy), so use a reasonable threshold
        if e_margin is None:
            # For binary classification, use 0.4 (keeps samples with entropy < 0.4, i.e., confident predictions)
            # This is equivalent to the ImageNet formula scaled for 2 classes
            e_margin = 0.4  # Reasonable threshold for binary classification
        
        self.num_samples_update_1 = 0  # After first filtering (reliability)
        self.num_samples_update_2 = 0  # After second filtering (reliability + redundancy)
        self.e_margin = e_margin  # Entropy threshold (Eqn. 3)
        self.d_margin = d_margin  # Cosine similarity threshold (Eqn. 5)

        self.current_model_probs = None  # Moving average of prob vector (Eqn. 4)

        self.fishers = fishers  # Fisher regularizer for anti-forgetting (Eqn. 9)
        self.fisher_alpha = fisher_alpha  # Trade-off β (Eqn. 8)

        # Save initial state
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Adaptation tracking
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,  # Samples that passed both filters
            'reliable_samples': 0,  # Samples that passed entropy filter
            'total_updates': 0,
        }

    def forward(self, input_ids=None, attention_mask=None, special_tokens_mask=None,
                mode="test", original_text=None, current_batch_num=None, **kwargs):
        """Forward pass with EATA adaptation for text inputs."""
        if self.episodic:
            self.reset()
        
        batch_size = input_ids.size(0) if input_ids is not None else 1
        self.adaptation_stats['total_samples'] += batch_size
        
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, proto_dist, proto_val, similarity, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(
                    model=self.model,
                    optimizer=self.optimizer,
                    fishers=self.fishers,
                    e_margin=self.e_margin,
                    current_model_probs=self.current_model_probs,
                    fisher_alpha=self.fisher_alpha,
                    num_samples_update=self.num_samples_update_2,
                    d_margin=self.d_margin,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    mode=mode,
                    original_text=original_text,
                    current_batch_num=current_batch_num,
                    **kwargs
                )
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.adaptation_stats['adapted_samples'] += num_counts_2
                self.adaptation_stats['reliable_samples'] += num_counts_1
                self.adaptation_stats['total_updates'] += num_counts_2
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs, proto_dist, proto_val = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    mode=mode,
                    original_text=original_text,
                    current_batch_num=current_batch_num,
                    **kwargs
                )

        return outputs, proto_dist, proto_val, similarity

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)

    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temperature = 1
    x = x / temperature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()
def forward_and_adapt_eata(model, optimizer, fishers, e_margin, current_model_probs,
                           fisher_alpha=50.0, d_margin=0.05, scale_factor=2,
                           num_samples_update=0, input_ids=None, attention_mask=None,
                           special_tokens_mask=None, mode="test", original_text=None,
                           current_batch_num=None, **kwargs):
    """Forward and adapt model on batch of text data with EATA filtering.
    
    Returns:
        outputs: Model predictions
        proto_dist: Prototype distances
        proto_val: Prototype values
        num_counts_2: Number of samples passing both filters
        num_counts_1: Number of samples passing first filter (reliability)
        updated_probs: Updated moving average probabilities
    """
    # Forward
    # ProtoLens now returns: (logits, loss_mu, augmented_loss, similarity)
    outputs, proto_dist, proto_val, similarity = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        mode=mode,
        original_text=original_text,
        current_batch_num=current_batch_num,
        **kwargs
    )
    
    # Compute entropy
    entropys = softmax_entropy(outputs)
    
    # Filter 1: Keep reliable samples (low entropy = confident predictions)
    # Keep samples where entropy < e_margin (confident predictions)
    filter_ids_1 = torch.where(entropys < e_margin)
    entropys_filtered = entropys[filter_ids_1]
    
    # Filter 2: Remove redundant samples (cosine similarity)
    if current_model_probs is not None and entropys_filtered.size(0) > 0:
        cosine_similarities = F.cosine_similarity(
            current_model_probs.unsqueeze(dim=0),
            outputs[filter_ids_1].softmax(1),
            dim=1
        )
        # Keep samples that are NOT redundant (cosine similarity < d_margin means different)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys_final = entropys_filtered[filter_ids_2]
        updated_probs = update_model_probs(
            current_model_probs,
            outputs[filter_ids_1][filter_ids_2].softmax(1),
            num_samples_update
        )
        num_counts_2 = entropys_final.size(0)
    else:
        entropys_final = entropys_filtered
        updated_probs = update_model_probs(
            current_model_probs,
            outputs[filter_ids_1].softmax(1),
            num_samples_update
        )
        num_counts_2 = entropys_final.size(0)
    
    # Compute reweighted loss
    if entropys_final.size(0) > 0:
        coeff = 1 / (torch.exp(entropys_final.clone().detach() - e_margin) + 1e-8)
        entropys_weighted = entropys_final.mul(coeff)
        loss = entropys_weighted.mean(0)
    else:
        # No samples passed filters, skip adaptation
        loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)
    
    # Add Fisher regularization if available
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (
                    fishers[name][0] * (param - fishers[name][1])**2
                ).sum()
        loss += ewc_loss
    
    if entropys_final.size(0) > 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    
    num_counts_1 = filter_ids_1[0].size(0)  # Samples passing entropy filter
    return outputs, proto_dist, proto_val, similarity, num_counts_2, num_counts_1, updated_probs


def update_model_probs(current_model_probs, new_probs, num_samples):
    """Update moving average of model probabilities."""
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return (current_model_probs * num_samples + 
                       new_probs.sum(0)) / (num_samples + new_probs.size(0))


def copy_model_and_optimizer(model, optimizer):
    """Copy model and optimizer states."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore model and optimizer states."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
