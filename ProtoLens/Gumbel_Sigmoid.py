"""
Gumbel-Sigmoid implementation for differentiable binary sampling.
Used in ProtoLens for soft masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSigmoid(nn.Module):
    """
    Gumbel-Sigmoid for differentiable binary sampling.
    """
    def __init__(self, temperature=1.0):
        super(GumbelSigmoid, self).__init__()
        self.temperature = temperature
    
    def sample_gumbel(self, shape, device='cuda', eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def forward(self, logits, Log=False, mask=None, return_soft=False, hard=False):
        """
        Apply Gumbel-Sigmoid to logits.
        
        Args:
            logits: Input logits
            Log: Whether to log (for debugging)
            mask: Optional mask to apply
            return_soft: Whether to return soft probabilities
            hard: Whether to use hard sampling (straight-through)
        
        Returns:
            Gumbel-Sigmoid sampled values
        """
        if self.training or not hard:
            # Sample from Gumbel-Sigmoid
            gumbel_noise = self.sample_gumbel(logits.shape, device=logits.device)
            y = logits + gumbel_noise
            y = torch.sigmoid(y / self.temperature)
            
            if return_soft:
                return y
            
            # Straight-through estimator
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y
            
            if mask is not None:
                y = y * mask
            
            return y
        else:
            # Deterministic inference
            y = torch.sigmoid(logits / self.temperature)
            
            if return_soft:
                return y
            
            y = (y > 0.5).float()
            
            if mask is not None:
                y = y * mask
            
            return y
