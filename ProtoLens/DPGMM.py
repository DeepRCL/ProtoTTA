import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StickBreakingProcess(nn.Module):
    def __init__(self, n_components):
        super(StickBreakingProcess, self).__init__()
        self.n_components = n_components
        self.alpha = nn.Parameter(torch.Tensor([1.0]))  # Dirichlet Process prior strength

    def forward(self, logits):
        # Apply sigmoid to logits to get the nu values in (0,1)
        nu = torch.sigmoid(logits)
        # Compute pi using the stick-breaking process
        pi = torch.ones_like(nu)
        prod_term = torch.ones_like(nu[:, 0])  # Initialize the product term for the first component
        for k in range(self.n_components):
            pi[:, k] = nu[:, k] * prod_term  # nu_k * product of (1 - nu_j)
            prod_term = prod_term * (1 - nu[:, k])  # Update the product term for next component
        return pi


class MixtureDensityNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_components):
        super(MixtureDensityNetwork, self).__init__()
        self.n_components = n_components  # Maximum number of Gaussian components
        self.n_input = n_input
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.pi_logits = nn.Linear(n_hidden, n_components)
        self.mu = nn.Linear(n_hidden, n_components)
        self.sigma = nn.Linear(n_hidden, n_components)
        self.stick_breaking = StickBreakingProcess(n_components)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        pi_logits = self.pi_logits(h)
        pi = self.stick_breaking(pi_logits)
        mu = self.mu(h).sigmoid() * 512
        sigma = torch.exp(self.sigma(h))
        return pi, mu, sigma

    def loss(self, pi, mu, sigma, y, l1_lambda=1e-4):
        gauss = torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        prob = torch.sum(pi * gauss, dim=1)
        nll = -torch.log(prob + 1e-8)
        l1_reg = l1_lambda * torch.sum(torch.abs(pi))
        return torch.mean(nll) + l1_reg




