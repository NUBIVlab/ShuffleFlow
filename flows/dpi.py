"""
    DPI: Diffusion Probabilistic Image Inpainting.
    official implementation: https://github.com/junyanz/DPI.
"""
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from utils.torch import batched_sum

from . import BaseFlow
from .realnvp import RealNVP


class DPI(BaseFlow):
    def __init__(self, network_config: DictConfig, img_shape: List[int], **kwargs):
        super().__init__()
        self.network_config = network_config
        self.img_shape = img_shape
        self.n_pixels = np.prod(self.img_shape)
                
        if self.network_config.name == "realnvp":  
            self.flow = RealNVP(
                ndim=self.n_pixels, 
                n_flow=self.network_config.num_flows,
                affine=self.network_config.affine,
                seqfrac=self.network_config.seqfrac,
                permute=self.network_config.permute,
                batch_norm=self.network_config.batch_norm
            )
        else:
            raise NotImplementedError(f"Flow network {self.network_config.name} not implemented.")
        
        self.softplus = None
        if self.network_config.softplus:
            self.softplus = nn.Softplus()
            # self.log_scale_factor = nn.Parameter(torch.ones(1, ) * math.log(network.scale), requires_grad=True)
    
    def init_sample(self, num_samples: int):
        return torch.randn(num_samples, self.n_pixels).cuda()
    
    def forward(self, z: torch.Tensor):
        x, logdet = self.flow.reverse(z)
        
        if self.softplus is not None:
            z = self.softplus(z) # * torch.exp(self.log_scale_factor)
            logdet += batched_sum(z - self.softplus(z)) # + self.log_scale_factor
            
        x = x.reshape(z.shape[0], -1, *self.img_shape)
        
        return x, logdet
    
    
    def sample(self, z: torch.Tensor):
        x, _ = self.flow.reverse(z)
        
        if self.softplus is not None:
            z = self.softplus(z) # * torch.exp(self.log_scale_factor)
            
        x = x.reshape(z.shape[0], -1, *self.img_shape)
        
        return x
