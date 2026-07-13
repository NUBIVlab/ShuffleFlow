"""Forward operator for non-blind deblurring."""

from typing import Dict

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn import functional as F

from utils.motionblur import *
from utils.torch import *

from . import BaseOperator


class Deblur(BaseOperator):
    def __init__(self, psf: torch.Tensor=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.psf = psf
        
    @staticmethod
    def convolve(img: torch.Tensor, psf: torch.Tensor, mode: str='fft', double_pad: bool=False) -> torch.Tensor:
        """2D convolution of two images."""
        if mode == 'fft':
            if double_pad:
                img, psf = pad_double(img), pad_double(psf)
                return crop_half(fftshift(ifft2(fft2(ifftshift(img, dim=(-2,-1))) * fft2(ifftshift(psf, dim=(-2,-1)))), dim=(-2,-1)).real)
            else:
                return fftshift(ifft2(fft2(ifftshift(img, dim=(-2,-1))) * fft2(ifftshift(psf, dim=(-2,-1)))), dim=(-2,-1)).real
        elif mode == 'conv':
            img = F.pad(img.unsqueeze(0), ((psf.shape[-2] - 1) // 2, (psf.shape[-2] - 1) // 2, (psf.shape[-1] - 1) // 2, (psf.shape[-1] - 1) // 2), mode=pad)
            return F.conv2d(img, psf.unsqueeze(1), groups=3).squeeze(0)
        else:
            raise ValueError(f'Invalid convolution mode: {mode}')
    
    @staticmethod
    def get_gaussian_kernel(shape: tuple, sigma: float) -> torch.Tensor:
        """Generate a 2D Gaussian kernel.

        Args:
            size (`int`): Kernel size.
            sigma (`float`): Standard deviation.

        Returns:
            `torch.Tensor`: Gaussian kernel.
        """
        x = torch.linspace(-shape[0]//2+1/2, shape[0]//2-1/2, shape[0])
        y = torch.linspace(-shape[1]//2+1/2, shape[1]//2-1/2, shape[1])
        X, Y = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, shape[0], shape[1])
        return kernel
    
    @staticmethod
    def get_motion_kernel(shape: tuple, intensity: float) -> torch.Tensor:
        """Generate a 2D motion kernel.

        Args:
            size (`int`): Kernel size.
            intensity (`float`): Intensity.
        """
        psf = Kernel(size=(shape[-2]//4, shape[-1]//4), intensity=intensity).kernelMatrix
        psf = torch.from_numpy(psf).view(1, psf.shape[-2], psf.shape[-1])
        psf = F.pad(psf, (3*shape[-2]//8, 3*shape[-1]//8, 3*shape[-2]//8, 3*shape[-1]//8), mode='constant', value=0)
        psf /= psf.sum()
        return psf
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolve(x, self.psf)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return fftshift(ifft2(torch.conj(fft2(self.psf)) * fft2(y)).real, dim=(-2,-1))
    
    def load_parameters(self, data: Dict):
        self.psf = data['psf'].to(self.device)
        
    def get_scale(self, y: torch.Tensor) -> torch.Tensor:
        return torch.ones(1, device=y.device)