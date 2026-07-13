"""
    ShuffleFlow.
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from utils.torch import batched_sum, get_mgrid

from . import BaseFlow
from .realnvp import *
from .siren import SIREN


class ShuffleFlow(BaseFlow):
    def __init__(self, network_config: DictConfig, img_shape: List[int], **kwargs):
        super().__init__()
        self.img_shape = img_shape
        
        assert network_config.shuffle_flows > 0, "Number of shuffle flows must be greater than 0."
        
        # Shuffle coordinates.
        self.shuffle_shape = [1]
        grid_shape = [1]
        self.shuffle_channels, self.shuffle_dim = img_shape[0], img_shape[0]
        try:
            for n in self.img_shape[1:]:
                assert n % network_config.downsample == 0
                grid_shape.append(network_config.downsample)
                self.shuffle_shape.append(n // network_config.downsample)
                self.shuffle_channels *= grid_shape[-1]
                self.shuffle_dim *= self.shuffle_shape[-1]
            self.shuffle_shape[0] = self.shuffle_channels
        except:
            raise ValueError(f'Downsample factor {network_config.downsample} must divide image shape {self.img_shape} evenly.')
        self.register_buffer('shuffle_coords', get_mgrid(grid_shape[1:]).unsqueeze(0))
        self.shuffle_encoder = SIREN(
            in_features=len(self.img_shape)-1,
            out_features=network_config.embed_dim,
            hidden_layers=network_config.encoder.hidden_layers,
            hidden_features=network_config.encoder.hidden_features, 
            act_func=network_config.encoder.act_func,
            pos_encoding=network_config.encoder.pos_enc,
            n_freqs=network_config.encoder.num_freqs
        )
        self.shuffle_flow = CondRealNVP(
            ndim=self.shuffle_dim, 
            ndim_cond=network_config.embed_dim,
            n_flow=network_config.shuffle_flows,
            affine=network_config.affine,
            permute=network_config.permute,
            seqfrac=network_config.decoder.seqfrac,
            batch_norm=network_config.decoder.batch_norm,
        )
        self.pixel_shuffle = nn.PixelShuffle(network_config.downsample)
        
        self.base_flow, self.patch_flow = None, None
        self.patch_encoder = None
        if network_config.name == "realnvp":
            if network_config.base_flows > 0:
                self.base_flow = RealNVP(
                    ndim=self.shuffle_dim, 
                    n_flow=network_config.base_flows,
                    affine=network_config.affine,
                    permute=network_config.permute,
                    seqfrac=network_config.decoder.seqfrac,
                    batch_norm=network_config.decoder.batch_norm,
                )
            if network_config.patch_flows > 0:
                # Patch coordinates.
                self.patch_shape = [1]
                grid_shape = [1]
                self.patch_channels, self.patch_dim = img_shape[0], img_shape[0]
                try:
                    for n in self.img_shape[1:]:
                        assert n % network_config.downsample == 0
                        grid_shape.append(n // network_config.downsample)
                        self.patch_shape.append(network_config.downsample)
                        self.patch_channels *= grid_shape[-1]
                        self.patch_dim *= self.patch_shape[-1]
                    self.patch_shape[0] = self.patch_channels
                except:
                    raise ValueError(f'Downsample factor {network_config.downsample} must divide image shape {self.img_shape} evenly.')
                self.register_buffer('patch_coords', get_mgrid(grid_shape[1:]).unsqueeze(0))

                self.patch_encoder = SIREN(
                    in_features=len(self.img_shape)-1,
                    out_features=network_config.embed_dim,
                    hidden_layers=network_config.encoder.hidden_layers,
                    hidden_features=network_config.encoder.hidden_features, 
                    act_func=network_config.encoder.act_func,
                    pos_encoding=network_config.encoder.pos_enc,
                    n_freqs=network_config.encoder.num_freqs
                )
                self.patch_flow = CondRealNVP(
                    ndim=self.patch_dim, 
                    ndim_cond=network_config.embed_dim,
                    n_flow=network_config.patch_flows,
                    affine=network_config.affine,
                    permute=network_config.permute,
                    seqfrac=network_config.decoder.seqfrac,
                    batch_norm=network_config.decoder.batch_norm,
                )
                self.unfold = nn.Unfold(kernel_size=self.patch_shape[-2:], stride=self.patch_shape[-2:])
                self.fold = nn.Fold(output_size=self.img_shape[-2:], kernel_size=self.patch_shape[-2:], stride=self.patch_shape[-2:])   
        else:
            raise NotImplementedError(f"Flow network {network_config.name} not implemented.")
        
        self.use_sigmoid = network_config.sigmoid
        self.softplus = None
        if network_config.softplus:
            self.softplus = nn.Softplus()
            # self.log_scale_factor = nn.Parameter(torch.ones(1, ) * math.log(network.scale), requires_grad=True)
    
    def init_sample(self, num_samples: int):
        return torch.randn(num_samples, 1, self.shuffle_dim).cuda()
        # return torch.randn(num_samples, self.shuffle_channels, self.shuffle_dim).cuda()
    
    def forward(self, z: torch.Tensor):
        logdet = 0.0

        # Base flow.
        if self.base_flow is not None:
            z, logdet_base = self.base_flow.reverse(z.view(z.shape[0], -1))
            logdet += batched_sum(logdet_base.view(z.shape[0], 1, -1).repeat(1, self.shuffle_channels, 1))
        x = z.reshape(z.shape[0], 1, self.shuffle_dim).repeat(1, self.shuffle_channels, 1)

        # Shuffle flow.   
        embeddings = self.shuffle_encoder(self.shuffle_coords).repeat(z.shape[0],1,1)
        x, logdet_shuffle = self.shuffle_flow.reverse(x.view(z.shape[0]*self.shuffle_channels, -1), embeddings.view(z.shape[0]*self.shuffle_channels, -1))
        x = x.reshape(z.shape[0], -1, *self.shuffle_shape[-2:])
        logdet += batched_sum(logdet_shuffle.view(z.shape[0], self.shuffle_channels, -1))
        x = self.pixel_shuffle(x)
        
        # Patch flow.
        if self.patch_flow is not None:
            embeddings = self.patch_encoder(self.patch_coords).repeat(z.shape[0],1,1)
            x = self.unfold(x).permute(0,2,1)
            x, logdet_patch = self.patch_flow.reverse(x.reshape(z.shape[0]*self.patch_channels, -1), embeddings.reshape(z.shape[0]*self.patch_channels, -1))
            x = x.reshape(z.shape[0], -1, *self.patch_shape[-2:])
            logdet += batched_sum(logdet_patch.view(z.shape[0], self.patch_channels, -1))
            x = x.view(z.shape[0],1,*self.patch_shape).permute(0,1,3,4,2).reshape(z.shape[0], np.prod(self.patch_shape[-2:]), self.patch_channels)
            x = self.fold(x)
        
        if self.softplus is not None:
            x_pre_softplus = x
            x = self.softplus(x_pre_softplus)
            log_jac_sp = torch.sigmoid(x_pre_softplus).clamp(min=torch.finfo(x.dtype).tiny).log()
            logdet += batched_sum(log_jac_sp)
            
        if self.use_sigmoid:
            x, logdet_sigmoid = self.sigmoid(x)
            logdet += batched_sum(logdet_sigmoid)
        
        return x, logdet
        
    def sample(self, z: torch.Tensor):
        # Base flow.
        if self.base_flow is not None:
            z, _ = self.base_flow.reverse(z.view(z.shape[0], -1))
        x = z.reshape(z.shape[0], 1, self.shuffle_dim).repeat(1, self.shuffle_channels, 1)

        # Shuffle flow.   
        embeddings = self.shuffle_encoder(self.shuffle_coords).repeat(z.shape[0],1,1)
        x, _ = self.shuffle_flow.reverse(x.view(z.shape[0]*self.shuffle_channels, -1), embeddings.view(z.shape[0]*self.shuffle_channels, -1))
        x = x.reshape(z.shape[0], -1, *self.patch_shape[-2:])
        x = self.pixel_shuffle(x)
        
        # Patch flow.
        if self.patch_flow is not None:
            embeddings = self.patch_encoder(self.patch_coords).repeat(z.shape[0],1,1)
            x = self.unfold(x).permute(0,2,1)
            x, _ = self.patch_flow.reverse(x.reshape(z.shape[0]*self.patch_channels, -1), embeddings.reshape(z.shape[0]*self.patch_channels, -1))
            x = x.reshape(z.shape[0], -1, *self.patch_shape[-2:])
            x = x.view(z.shape[0],1,*self.patch_shape).permute(0,1,3,4,2).reshape(z.shape[0], np.prod(self.patch_shape[-2:]), self.patch_channels)
            x = self.fold(x)
        
        if self.softplus is not None:
            x = self.softplus(x) # * torch.exp(self.log_scale_factor)
        
        if self.use_sigmoid:
            x, _ = self.sigmoid(x)
            
        return x