from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossAdapt(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, sparsity_weight=0.0, clustering_weight=0.1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.sparsity_weight = sparsity_weight
        self.clustering_weight = clustering_weight
        # Cache initial state
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        print(f"Sparsity weight: {sparsity_weight}, Clustering weight: {clustering_weight}")

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            # We need access to the internal similarity scores 'values'
            # Assuming model returns: logits, min_distances, similarities
            outputs, min_distances, similarities = self.forward_and_adapt(x)

        return outputs, min_distances, similarities

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found in LossAdapt."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # 1. Forward Pass
        # outputs: (B, Num_Classes)
        # similarities: (B, Num_Prototypes, K_SubPrototypes) or similar
        logits, min_distances, similarities = self.model(x)

        # -----------------------------------------------------------
        # LOSS 1: Prototype Sparsity (Noise Suppression)
        # -----------------------------------------------------------
        # On noisy images, many prototypes activate weakly (e.g., 0.2).
        # We want the distribution to be sparse: mostly 0s, few high values.
        # L1 penalty on the similarities pushes weak noise to zero.
        # We clamp at 0 because ProtoViT sums similarities (assuming positive evidence).
        sim_relu = torch.relu(similarities) 
        loss_sparsity = sim_relu.mean()

        # -----------------------------------------------------------
        # LOSS 2: Class-Conditional Evidence Alignment
        # -----------------------------------------------------------
        # Identify the pseudo-label (what the model thinks it is currently)
        with torch.no_grad():
            pred_class = logits.argmax(dim=1) # (B,)
            
            # Identify which prototypes belong to which class
            # model.prototype_class_identity is usually (Num_Prototypes, Num_Classes)
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1) # (Num_Prototypes,)

        # Flatten similarities for masking: (B, Num_Prototypes)
        # Note: If ProtoViT returns (B, Num_Prototypes, K), max over K first to get "Prototype Score"
        if similarities.dim() == 3:
            # Take max over sub-prototypes (greedy match approximation)
            sim_scores, _ = similarities.max(dim=2) 
        else:
            sim_scores = similarities

        # Create a mask: 1 if prototype belongs to pred_class, 0 otherwise
        # (B, Num_Prototypes)
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()

        # Separation Objective:
        # 1. Pull features CLOSER to prototypes of the predicted class (Maximize Sim)
        # 2. Push features AWAY from prototypes of other classes (Minimize Sim)
        
        # We use the mask to split the objective
        # For target class: Minimize (1 - similarity) -> drives sim towards 1
        loss_target_cluster = (1.0 - sim_scores) * target_mask
        
        # For non-target class: Minimize (similarity) -> drives sim towards 0
        loss_nontarget_suppress = sim_scores * (1.0 - target_mask)

        # Average them
        loss_clustering = loss_target_cluster.sum() / (target_mask.sum() + 1e-7) + \
                          loss_nontarget_suppress.sum() / ((1-target_mask).sum() + 1e-7)

        # -----------------------------------------------------------
        # TOTAL LOSS
        # -----------------------------------------------------------
        # Balance the losses. Sparsity prevents hallucination. Clustering aligns features.
        # Suggested weights: Sparsity=0.1, Clustering=0.5
        loss = self.sparsity_weight * loss_sparsity + self.clustering_weight * loss_clustering

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, min_distances, similarities

# Helper functions
def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with LossAdapt adaptation."""
    # train mode, because LossAdapt optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what LossAdapt updates
    model.requires_grad_(False)
    # configure norm for LossAdapt updates: enable grad + force batch statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

