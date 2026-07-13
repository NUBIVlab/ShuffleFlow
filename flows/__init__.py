from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class BaseFlow(nn.Module):
    @abstractmethod
    def init_sample(self, num_samples: int):
        pass
    
    @abstractmethod
    def forward(self, z: torch.Tensor):
        pass
    
    @abstractmethod
    def sample(self, num_samples: int):
        pass
    
    def sigmoid(self, x: torch.Tensor):
        s = F.sigmoid(x)
        # Clamp so 2*s*(1-s) does not underflow to 0 before log (avoids -inf logdet).
        deriv = (2 * s * (1 - s)).clamp(min=torch.finfo(x.dtype).tiny)
        return 2 * s - 1, torch.log(deriv)
    
    
from .cf_nerf import CFNeRF
from .dpi import DPI
from .shuffleflow import ShuffleFlow
from .utils import *
