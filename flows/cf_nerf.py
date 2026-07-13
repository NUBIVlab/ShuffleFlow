"""
    Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty Quantification.
    official implementation: https://github.com/poetrywanderer/CF-NeRF.
"""
from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from utils.torch import *

from . import BaseFlow
from .flow_funcs import *
from .realnvp import ActNorm, Decoder
from .siren import SIREN


class ConditionalFlow(nn.Module):
    def __init__(self, flow_func: str):
        super().__init__()
        
        if flow_func == 'affine':
            self.flow = AffineFlow()
        elif flow_func == 'planar':
            self.flow = PlanarFlow()
        elif flow_func == 'radial':
            self.flow = RadialFlow()
        else:
            raise NotImplementedError("Flow function not supported.")
        self.act_norm = ActNorm()
    
    def forward(self, z, flow_params):
        z, log_det_norm = self.act_norm.forward(z)
        z, log_det_flow = self.flow.forward(z, flow_params) 
        return z, log_det_flow + log_det_norm
    
    def sample(self, z, flow_params):
        z = self.act_norm.sample(z)
        z = self.flow.sample(z, flow_params)
        return z
    

class CFNeRF(BaseFlow):
    def __init__(self, network_config: DictConfig, img_shape: List[int], **kwargs):
        super().__init__()
        self.img_shape = img_shape

        self.n_dims = len(img_shape) - 1
        if network_config.flow_func == 'affine':
            self.num_params = 2
        elif network_config.flow_func == 'planar':
            self.num_params = 3
        elif network_config.flow_func == 'radial':
            self.num_params = 3
        else:
            raise NotImplementedError("Flow function not supported.")
        
        self.register_buffer('coords', get_mgrid(self.img_shape[-2:]).unsqueeze(0))
        
        self.encoder = SIREN(
            in_features=len(self.img_shape)-1,
            out_features=network_config.embed_dim,
            hidden_layers=network_config.encoder.hidden_layers,
            hidden_features=network_config.encoder.hidden_features, 
            act_func=network_config.encoder.act_func,
            pos_encoding=network_config.encoder.pos_enc,
            n_freqs=network_config.encoder.num_freqs
        )
        
        decoders, flows = [], []
        for idx in range(network_config.num_flows):
            num_params = self.num_params if idx < network_config.num_flows-1 else 2
            flow_func = network_config.flow_func if idx < network_config.num_flows-1 else 'affine'
            decoders.append(Decoder(network_config.embed_dim, num_params, network_config.decoder.hidden_layers, network_config.decoder.hidden_features, network_config.decoder.batch_norm))
            flows.append(ConditionalFlow(flow_func))
        self.decoders = nn.ModuleList(decoders)
        self.flows = nn.ModuleList(flows)
        # self.decoders = nn.ModuleList([
        #     *[Decoder(network_config.embed_dim, self.num_params, network_config.decoder.hidden_layers, network_config.decoder.hidden_features, network_config.decoder.batch_norm) for _ in range(network_config.num_flows-1)],
        #      Decoder(network_config.embed_dim, 2, network_config.decoder.hidden_layers, network_config.decoder.hidden_features, network_config.decoder.batch_norm) # Last flow should be affine.
        # ])
        # self.flows = nn.ModuleList([
        #     *[ConditionalFlow(network_config.flow_func) for _ in range(network_config.num_flows-1)], 
        #     ConditionalFlow('affine') # Last flow should be affine.
        # ])
        
        self.softplus = None
        if network_config.softplus:
            self.softplus = nn.Softplus()
            # self.log_scale_factor = nn.Parameter(torch.ones(1, ) * math.log(network.scale), requires_grad=True)

    def init_sample(self, n: int):
        return torch.randn(n, self.coords.shape[1], 1).cuda()
    
    def forward(self, z: torch.Tensor):
        log_det_J = 0.0
        embeddings = self.encoder(self.coords.view(-1, self.n_dims))

        for decoder, flow in zip(self.decoders, self.flows):
            params = decoder(embeddings).squeeze(1)
            params = params.unsqueeze(0).repeat(z.shape[0],1,1)
            z, log_det_Ji = flow.forward(z, params)
            log_det_J += log_det_Ji
            
        if self.softplus is not None:
            z = self.softplus(z) # * torch.exp(self.log_scale_factor)
            log_det_J += z - self.softplus(z) # + self.log_scale_factor
        
        return z, log_det_J
    
    def sample(self, z: torch.Tensor):
        embeddings = self.encoder(self.coords.view(-1, self.n_dims))

        for decoder, flow in zip(self.decoders, self.flows):
            params = decoder(embeddings).squeeze(1)
            params = params.unsqueeze(0).repeat(z.shape[0],1,1)
            z = flow.sample(z, params)
            
        if self.softplus is not None:
            z = self.softplus(z) # * torch.exp(self.log_scale_factor)
            
        return z
