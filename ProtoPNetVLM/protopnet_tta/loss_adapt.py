"""
MEMO (Marginal Entropy Minimization with One test point) implementation.
Adapted from ProtoViT TTA implementation for ProtoPNet.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossAdapt(nn.Module):
    """MEMO: Test-time adaptation using marginal entropy minimization."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        # Cache initial state
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Adaptation tracking statistics
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'total_updates': 0,
        }

    def forward(self, x):
        if self.episodic:
            self.reset()

        # Track adaptation
        batch_size = x.size(0)
        self.adaptation_stats['total_samples'] += batch_size
        self.adaptation_stats['adapted_samples'] += batch_size

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
            self.adaptation_stats['total_updates'] += 1

        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt using marginal entropy."""
        # Forward Pass
        outputs = self.model(x)

        # Extract logits
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute marginal entropy loss
        # Marginal entropy = H(E[p(y|x)]) where expectation is over the batch
        probs = F.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)  # Average over batch

        # Entropy of the average distribution
        marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()

        # MEMO minimizes marginal entropy
        loss = marginal_entropy

        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return outputs


# Helper functions
def collect_params(model, adaptation_mode='batchnorm_addon'):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    """
    params = []
    names = []

    # Get the core model (handle ProtoPNet wrapper)
    core = model.core if hasattr(model, 'core') else model

    # For ProtoPNet: collect from features and add_on_layers
    if hasattr(core, 'features') and hasattr(core, 'add_on_layers'):
        for nm, m in core.features.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"features.{nm}.{np_name}")

        for nm, m in core.add_on_layers.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"add_on_layers.{nm}.{np_name}")
    else:
        # Generic normalization collection
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np_name}")

    return params, names


def configure_model(model):
    """Configure model for use with MEMO adaptation."""
    # train mode, because MEMO optimizes the model to minimize marginal entropy
    model.train()
    # disable grad, to (re-)enable only what MEMO updates
    model.requires_grad_(False)

    # Get the core model (handle ProtoPNet wrapper)
    core = model.core if hasattr(model, 'core') else model

    # configure norm for MEMO updates: enable grad + force batch statistics
    if hasattr(core, 'features'):
        for m in core.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

    if hasattr(core, 'add_on_layers'):
        for m in core.add_on_layers.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

    # For generic models without features/add_on_layers structure
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
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
