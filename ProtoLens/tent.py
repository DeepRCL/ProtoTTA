"""
TENT adaptation for ProtoLens text classification models.
Adapted from ProtoViT's tent.py for text domain.
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

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Adaptation tracking statistics
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,  # Tent adapts ALL samples
            'total_updates': 0,    # Total optimizer steps
        }

    def forward(self, input_ids=None, attention_mask=None, special_tokens_mask=None, 
                mode="test", original_text=None, current_batch_num=None, **kwargs):
        """Forward pass with adaptation for text inputs."""
        if self.episodic:
            self.reset()
        
        # Track adaptation
        batch_size = input_ids.size(0) if input_ids is not None else 1
        self.adaptation_stats['total_samples'] += batch_size
        self.adaptation_stats['adapted_samples'] += batch_size  # Tent adapts ALL samples
        
        for _ in range(self.steps):
            outputs = forward_and_adapt_text(
                self.model, self.optimizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode=mode,
                original_text=original_text,
                current_batch_num=current_batch_num,
                **kwargs
            )
            # One update per sample per step
            self.adaptation_stats['total_updates'] += batch_size

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        # Only reset model parameters, NOT optimizer state
        # Preserves Adam momentum/variance for effective episodic updates
        self.model.load_state_dict(self.model_state, strict=True)

    def forward_no_adapt(self, input_ids=None, attention_mask=None, special_tokens_mask=None,
                         mode="test", original_text=None, current_batch_num=None, **kwargs):
        """Forward pass without adaptation (used for metrics)."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            mode=mode,
            original_text=original_text,
            current_batch_num=current_batch_num,
            **kwargs
        )
    
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
def forward_and_adapt_text(model, optimizer, input_ids=None, attention_mask=None,
                           special_tokens_mask=None, mode="test", original_text=None,
                           current_batch_num=None, **kwargs):
    """Forward and adapt model on batch of text data.
    
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    # ProtoLens now returns: (logits, loss_mu, augmented_loss, similarity)
    outputs, proto_distances, proto_values, similarity = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        mode=mode,
        original_text=original_text,
        current_batch_num=current_batch_num,
        **kwargs
    )
    
    # adapt - minimize entropy
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return outputs, proto_distances, proto_values, similarity


def collect_params(model):
    """Collect the affine scale + shift parameters from normalization layers.
    
    Walk the model's modules and collect all batch/layer normalization parameters.
    Return the parameters and their names.
    
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatibility with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)) 
                    for m in model.modules()])
    assert has_norm, "tent needs normalization for its optimization"
