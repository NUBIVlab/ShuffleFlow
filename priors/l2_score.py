import torch
from torch import nn

from utils.sde import *


class L2Score(nn.Module):
    def __init__(self, weight, sigma, model, **kwargs):
        super().__init__()
        self.weight = weight
        self.sigma = sigma
        self.model = model

    def forward(self, x, epoch=None):
        x_noisy = x + torch.randn_like(x) * self.sigma
        score = self.model(x_noisy, torch.as_tensor([self.sigma]).to(x.device))
        loss = score.square().flatten(start_dim=1).sum(dim=1)
        return self.weight * loss
