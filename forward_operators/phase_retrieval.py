"""Forward operator for Fourier phase retrieval."""

from typing import Dict

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn import functional as F

from utils.phase_retrieval import *
from utils.torch import *

from . import BaseOperator


class PhaseRetrieval(BaseOperator):
    def __init__(self, oversample: float=4.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.oversample = oversample
        
    @staticmethod
    def image_to_amplitude(x: torch.Tensor, oversample: float=4.0) -> torch.Tensor:
        x = x.to(torch.float64)
        pad = get_pad(oversample, x.shape[-1])
        x_pad = F.pad(x, (pad, pad, pad, pad))
        amplitude = ifftshift(fft2(fftshift(x_pad, dim=(-2,-1)), norm='ortho'), dim=(-2,-1)).abs()
        return amplitude

    @staticmethod
    def amplitude_to_image(amplitude: torch.Tensor, crop: int=0) -> torch.Tensor:
        img = ifftshift(ifft2(fftshift(amplitude, dim=(-2,-1)), norm='ortho'), dim=(-2,-1)).real
        return img[..., crop:-crop, crop:-crop] if crop > 0 else img
    
    def forward(self, x, **kwargs):
        return self.image_to_amplitude(x, oversample=self.oversample)
    
    def load_parameters(self, data: Dict):
        pass