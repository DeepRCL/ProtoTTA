from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tent import collect_params, configure_model, \
    copy_model_and_optimizer, load_model_and_optimizer


class FisherProto(nn.Module):
    """
    Fisher-guided prototype adaptation.

    - Uses prototype sparsity + class-conditional clustering (no Tent/entropy).
    - Maintains an online estimate of Fisher information per prototype based on
      the squared gradient of the loss w.r.t. prototype similarities.
    - Weights the loss terms by Fisher scores so that highly informative
      prototypes drive adaptation more than noisy ones.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 fisher_momentum: float = 0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "FisherProto requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Cache initial state (for episodic adaptation)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Per-prototype Fisher scores (initialized to uniform after we see model)
        num_prototypes = self.model.prototype_class_identity.size(0)
        fisher_init = torch.zeros(num_prototypes, dtype=torch.float32)
        self.register_buffer("fisher_scores", fisher_init)
        self.fisher_momentum = fisher_momentum

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, min_distances, similarities = self.forward_and_adapt(x)

        return outputs, min_distances, similarities

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found here."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        One adaptation step:
        - Forward pass to get logits and prototype similarities.
        - Compute sparsity and clustering losses per prototype.
        - Weight losses by Fisher scores (importance weighting).
        - Update Fisher scores from gradients w.r.t. similarities.
        - Take an optimizer step on the configured parameters (BN/LayerNorm).
        """
        # 1) Forward through ProtoViT
        logits, min_distances, similarities = self.model(x)

        # We need gradients at this intermediate tensor to estimate Fisher.
        similarities.retain_grad()

        # -----------------------------------------------------------
        # Prepare Fisher weights for prototypes
        # -----------------------------------------------------------
        # similarities: (B, P, K) or (B, P)
        if similarities.dim() == 3:
            # Aggregate sub-prototypes into a prototype-level score
            sim_scores, _ = similarities.max(dim=2)  # (B, P)
        else:
            sim_scores = similarities  # (B, P)

        # If Fisher scores are all zeros (beginning of adaptation),
        # fall back to uniform weights.
        with torch.no_grad():
            if self.fisher_scores.sum() <= 0:
                fisher_weights = torch.ones_like(self.fisher_scores)
            else:
                fisher_weights = self.fisher_scores.clamp(min=0.0)
            
            # Normalize Fisher weights to sum to 1 for proper weighting
            # This ensures prototypes with higher Fisher info get more attention
            fisher_weights = fisher_weights / (fisher_weights.sum() + 1e-8)  # (P,)

        # -----------------------------------------------------------
        # LOSS 1: Fisher-weighted Prototype Sparsity
        # -----------------------------------------------------------
        # Compute sparsity loss per prototype, then weight by Fisher
        if similarities.dim() == 3:
            sim_relu = torch.relu(similarities)  # (B, P, K)
            # Average over batch and sub-prototypes: (B, P, K) -> (P,)
            loss_per_proto_sparsity = sim_relu.mean(dim=(0, 2))  # (P,)
        else:
            sim_relu = torch.relu(similarities)  # (B, P)
            # Average over batch: (B, P) -> (P,)
            loss_per_proto_sparsity = sim_relu.mean(dim=0)  # (P,)
        
        # Weight the loss by Fisher importance, then sum
        loss_sparsity = (loss_per_proto_sparsity * fisher_weights).sum()

        # -----------------------------------------------------------
        # LOSS 2: Fisher-weighted Class-Conditional Clustering
        # -----------------------------------------------------------
        with torch.no_grad():
            # Pseudo-labels from current model prediction
            pred_class = logits.argmax(dim=1)  # (B,)

            # Prototype class assignment
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1)  # (P,)

        # (B, P): 1 if prototype belongs to predicted class
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()

        # [FIX] Use sum(dim=0) instead of mean(dim=0) to avoid double normalization
        # We want the total error first, then we divide by the count of active samples.
        # This gives us the true average error per occurrence in the batch.
        loss_target_sum = ((1.0 - sim_scores) * target_mask).sum(dim=0)  # (P,)
        loss_nontarget_sum = (sim_scores * (1.0 - target_mask)).sum(dim=0)  # (P,)
        
        # Normalize by count (average error per occurrence in the batch)
        target_count = target_mask.sum(dim=0) + 1e-7  # (P,)
        nontarget_count = (1.0 - target_mask).sum(dim=0) + 1e-7  # (P,)
        
        loss_target_norm = loss_target_sum / target_count  # (P,)
        loss_nontarget_norm = loss_nontarget_sum / nontarget_count  # (P,)
        
        # Combine target and non-target losses per prototype
        loss_per_proto_clustering = loss_target_norm + loss_nontarget_norm  # (P,)
        
        # Weight the clustering loss by Fisher importance, then sum
        loss_clustering = (loss_per_proto_clustering * fisher_weights).sum()

        # -----------------------------------------------------------
        # TOTAL LOSS
        # -----------------------------------------------------------
        # Sparsity and clustering losses are weighted by Fisher information.
        # This ensures that prototypes with higher Fisher scores (more informative)
        # contribute more to the adaptation objective.
        loss = 0.25 * loss_sparsity + 0.5 * loss_clustering

        loss.backward()

        # -----------------------------------------------------------
        # Update Fisher scores from gradients w.r.t. similarities
        # -----------------------------------------------------------
        with torch.no_grad():
            grad_sim = similarities.grad  # same shape as similarities
            if grad_sim is not None:
                if grad_sim.dim() == 3:
                    # Average over batch and sub-prototypes: (B, P, K) -> (P,)
                    fisher_batch = grad_sim.pow(2).mean(dim=(0, 2))
                else:
                    # (B, P) -> (P,)
                    fisher_batch = grad_sim.pow(2).mean(dim=0)

                # EMA update of Fisher scores
                self.fisher_scores.mul_(self.fisher_momentum).add_(
                    fisher_batch * (1.0 - self.fisher_momentum)
                )

        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, min_distances, similarities


