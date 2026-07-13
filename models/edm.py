"""Preconditioning module for EDM.

"""

import torch
from torch import nn


class VPPrecondition(nn.Module):
    def __init__(self,
        model: nn.Module,                          
        mode: str = 'vp',
        beta_d: float = 19.9,         
        beta_min: float = 0.1,        
        N: int = 1000,
        epsilon_t: float = 1e-3,
        **kwargs
    ):
        super().__init__()
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.N = N
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = model

    def forward(self, x, sigma):
        c_skip = 1
        c_out = - sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.N - 1) * self.sigma_inv(sigma)
        
        t = torch.tensor([c_noise]).repeat(x.shape[0]).view(-1,)
        return c_skip * x + c_out * self.model(c_in * x, t)

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

