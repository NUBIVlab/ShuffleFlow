import math
from typing import Dict

import torch
from omegaconf import DictConfig
from torch import nn

from forward_operators import BaseOperator
from utils.torch import *


def alpha_divergence(data_likelihood: torch.Tensor, regularization: torch.Tensor, entropy: torch.Tensor, alpha: float):
    if alpha == 1.0: # KL Divergence.
        return data_likelihood.mean() + regularization.mean() + entropy.mean()
    else: # Alpha Divergence.
        # print(data_likelihood.shape, regularization.shape, entropy.shape)
        return torch.logsumexp((data_likelihood+regularization+entropy)*(alpha-1), dim=0) / (alpha-1) - math.log(data_likelihood.shape[0]) / (alpha - 1)
        # return torch.log(1e-10+torch.mean(torch.exp((data_likelihood+regularization+entropy) * (alpha-1)))) / (alpha-1)
    
    
class ForwardWrapper(nn.Module):
    def __init__(self, loss_config: DictConfig, prior_config: DictConfig, data_config: DictConfig, flow: nn.Module, forward_operator: BaseOperator, prior: nn.Module, measurement: torch.Tensor=None):
        super().__init__()
        self.loss_config = loss_config
        self.prior_config = prior_config
        self.data_config = data_config
        self.flow = flow
        self.forward_operator = forward_operator
        self.prior = prior
        # # Register measurement as buffer to ensure proper device handling with DataParallel
        # self.register_buffer('measurement', measurement)
        # self.scale = self.forward_operator.get_scale(self.measurement, normalized=data_config.normalize).item()
        
    def forward(self, z: torch.Tensor, data: Dict,epoch: int):
        # Draw samples.
        samples, logdet = self.flow(z)
        samples = samples.view(-1, self.data_config.num_channels, self.data_config.img_size, self.data_config.img_size)
        
        # Compute loss components.
        likelihood = self.forward_operator.loss(self.forward_operator.unnormalize(samples), data['measurement']) / self.loss_config.sigma ** 2
        prior = self.prior(samples.float(), epoch=epoch)
        entropy = - self.loss_config.beta * batched_sum(logdet)
        
        # Annealing.
        weight = min(10**(- self.loss_config.starting_order + epoch/self.loss_config.decay_rate), 1.0)
        if self.loss_config.anneal in ['llh', 'both']:
            likelihood = weight * likelihood
        if self.loss_config.anneal in ['prior', 'both']:
            prior = weight * prior
        if self.loss_config.anneal == 'entropy':
            entropy = weight * entropy
        return likelihood, prior, entropy
