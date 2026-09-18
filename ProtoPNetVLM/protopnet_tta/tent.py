"""
Tent: Fully test-time adaptation by entropy minimization.

Adapted from ProtoViT TTA implementation for ProtoPNet.
Based on: https://arxiv.org/abs/2006.10726
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Save state for reset
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
            outputs = forward_and_adapt(x, self.model, self.optimizer)
            self.adaptation_stats['total_updates'] += batch_size

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)

    def forward_no_adapt(self, x):
        """Forward pass without adaptation (used for metrics)."""
        return self.model(x)

    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found in Tent."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    Includes diversity regularization to prevent mode collapse (IM Loss).
    """
    # forward - handle ProtoPNet output format (logits, min_distances)
    outputs = model(x)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # If no optimizer, just return (no adaptation possible)
    if optimizer is None:
        return outputs

    # adapt
    # 1. Entropy minimization H(Y|X)
    probs = logits.softmax(1)
    # Use log_softmax for numerical stability
    entropy_loss = softmax_entropy(logits).mean(0)

    # 2. Diversity maximization -H(Y) (Information Maximization)
    # Enforce uniform marginal distribution over the batch
    # This prevents the model from collapsing to predicting a single class
    marginal = probs.mean(0)
    marginal_entropy = -(marginal * torch.log(marginal + 1e-6)).sum()

    # Total loss: Minimize entropy + Maximize marginal entropy
    # Weight of 1.0 is standard for IM
    loss = entropy_loss - 1.0 * marginal_entropy

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model, adaptation_mode='batchnorm_addon'):
    """Collect parameters for adaptation.

    Following the ProtoViT approach: adapt BatchNorm parameters + add_on_layers.
    This gives much better TTA performance than only adapting add_on_layers.

    Args:
        model: The model to collect parameters from
        adaptation_mode:
            'batchnorm_addon' - BatchNorm weights/biases + add_on_layers (default, recommended)
            'batchnorm_only' - Only BatchNorm layers
            'addon_only' - Only add_on_layers Conv weights
            'full' - BatchNorm + add_on_layers + prototype_vectors

    Returns:
        params: List of parameters to adapt
        names: List of parameter names
    """
    params = []
    names = []

    # Get the core model if wrapped
    if hasattr(model, 'core'):
        core = model.core
        prefix = 'core.'
    else:
        core = model
        prefix = ''

    # Collect BatchNorm/LayerNorm parameters
    if adaptation_mode in ['batchnorm_addon', 'batchnorm_only', 'full']:
        # Collect from features (VGG backbone)
        for nm, m in core.features.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}features.{nm}.{np_name}")

        # Collect from add_on_layers (if they have BatchNorm, which they usually don't)
        for nm, m in core.add_on_layers.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}add_on_layers.{nm}.{np_name}")

    # Collect add_on_layers Conv parameters
    if adaptation_mode in ['batchnorm_addon', 'addon_only', 'full']:
        for nm, p in core.add_on_layers.named_parameters():
            if p.requires_grad and nm not in [n.split('.')[-1] for n in names]:
                params.append(p)
                names.append(f"{prefix}add_on_layers.{nm}")

    # Collect prototype vectors
    if adaptation_mode == 'full':
        if hasattr(core, 'prototype_vectors') and core.prototype_vectors.requires_grad:
            params.append(core.prototype_vectors)
            names.append(f"{prefix}prototype_vectors")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict()) if optimizer else None
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)


def configure_model(model, adaptation_mode='batchnorm_addon'):
    """Configure model for Tent adaptation.

    Following ProtoViT: Use train mode for BatchNorm layers with track_running_stats=False
    to force them to use batch statistics. This is more effective than eval mode.

    Args:
        adaptation_mode: Same as collect_params
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
    if adaptation_mode in ['batchnorm_addon', 'batchnorm_only', 'full']:
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
    if adaptation_mode in ['batchnorm_addon', 'addon_only', 'full']:
        for p in core.add_on_layers.parameters():
            p.requires_grad = True

    # Enable prototype vectors
    if adaptation_mode == 'full':
        if hasattr(core, 'prototype_vectors'):
            core.prototype_vectors.requires_grad = True

    return model


def check_model(model):
    """Check model for compatibility with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: check which require grad"
    assert not has_all_params, "tent should not update all params: check which require grad"
