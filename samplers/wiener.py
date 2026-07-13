"""
    Wiener deconvolution.
"""

import torch
import torch.nn as nn
import tqdm
from torch.fft import fft2, fftshift, ifft2, ifftshift

from priors.tv import TotalVariation

from . import BaseSampler


class Wiener(BaseSampler):
    def __init__(
        self,
        img_shape,
        lam,
        forward_op,
        **kwargs,
    ):
        super(Wiener, self).__init__(img_shape, None, forward_op)
        self.lam = lam

    def __call__(self, observation, num_samples=1, verbose=True):
        H = fft2(ifftshift(self.forward_op.psf, dim=(-2,-1)))
        Ht = torch.conj(H)
        Y = fft2(ifftshift(observation, dim=(-2,-1)))
        X = Ht * Y / (Ht * H + self.lam)
        x = fftshift(ifft2(X, dim=(-2,-1)), dim=(-2,-1)).real
        return self.forward_op.normalize(x)
