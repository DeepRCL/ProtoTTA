
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from Gumbel_Sigmoid import *
from utils import *

class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class AdaptiveMask(nn.Module):
    def __init__(self, ramp_size=3, init_val=0.001, shape=(1,)):
        nn.Module.__init__(self)
        self._ramp_size = ramp_size
        self.max_length = 512
        self.batch_size = 16
        self._loss_coeff = 0.01
        self.current_val_left = init_val # nn.Parameter(torch.zeros(*shape) + init_val)
        self.current_val_right = init_val # nn.Parameter(torch.zeros(*shape) + init_val)


    def forward(self, all_selected_token_index, sigma, pi,candidates=None, batch_size=16, num_prototypes=20): # (16, 512, 20)
        # pi_active, all_selected_token_index, sigma = remove_deactivated_elements(all_selected_token_index, sigma, pi)
        all_mask = []

        for max_size_all,sigma_curr,pi_curr in zip(all_selected_token_index, sigma, pi):
            # logging.info(max_size_all)
            new_sigma = []
            new_max_size_all = []
            threshold = torch.mean(pi_curr)
            for i, ele in enumerate(pi_curr):
                if ele >= threshold or ele == torch.max(pi_curr):
                    new_sigma.append(sigma_curr[i])
                    new_max_size_all.append(max_size_all[i])
            # logging.info(new_max_size_all)
            if len(new_max_size_all) == 0:
                sigma_curr = sigma_curr
                max_size_all = max_size_all
            else:
                sigma_curr = new_sigma
                max_size_all = new_max_size_all
            max_size_all = [torch.clamp(max_size, min=1.0) for max_size in max_size_all]


            max_size_all = [torch.round(i.clamp(1.0, 511.0)) for i in max_size_all]
            mask_left = [torch.linspace((1 - max_size.item()), 0, steps=int(max_size.item())).unsqueeze(0).cuda() + self.current_val_left* sigma_ * max_size for max_size,sigma_ in zip(max_size_all, sigma_curr)]  
            mask_right = [torch.linspace(-1,  (max_size.item() - self.max_length - 1), steps=self.max_length - int(max_size.item())).unsqueeze(0).cuda() + self.current_val_right * sigma_ * (self.max_length - max_size) for max_size,sigma_ in zip(max_size_all, sigma_curr)]
            mask = torch.vstack([torch.cat((mask_left[i], mask_right[i]), 1) for i in range(len(max_size_all))])#.reshape(batch_size, -1, num_prototypes)
            mask = mask / self._ramp_size + 1
            # mask = mask.clamp(0, 1)
            mask = torch.sum(mask, dim=0)
            mask = mask.clamp(0, 1)
            mask = Binarize.apply(mask)
            all_mask.append(mask)
        mask = torch.vstack(all_mask)
        mask = mask.clamp(0, 1)
        return mask

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        """this need to be called after each update"""
        self.current_val_left.data.clamp_(0.001, 0.5)
        self.current_val_right.data.clamp_(0.001, 0.5)
        
    def get_loss(self):
        """a loss term for regularizing the span length"""
        # print(self.current_val_left)
        return self._loss_coeff * self.max_length * (self.current_val_left+self.current_val_right).mean()
