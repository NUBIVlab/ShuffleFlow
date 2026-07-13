"""Helper functions for MRI."""

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift


def image_to_kspace(x: torch.Tensor, mask: torch.Tensor=None, noise_ratio: float=0.0) -> torch.Tensor:
    """Convert image space data to k-space using FFT.

    Args:
        x (Tensor): Input image data of shape `(..., Nx, Ny)`
        mask (Tensor, optional): Mask to apply in k-space. Defaults to `None`.

    Returns:
        Tensor: k-space data of shape `(..., Nx, Ny)`.
    """
    kspace = ifftshift(fft2(fftshift(x, dim=(-2,-1))), dim=(-2,-1))
    sigma = kspace.abs().max() * noise_ratio
    # noise = (1.0+1.0j) * torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * sigma/math.sqrt(2))
    noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * sigma) if noise_ratio > 0 else 0.0
    if mask is not None:
        return (kspace + noise) * mask
    return kspace + noise


def kspace_to_image(k: torch.Tensor) -> torch.Tensor:
    """Convert k-space data to image space using IFFT.

    Args:
        k (Tensor): Input k-space data of shape `(..., Nx, Ny)`

    Returns:
        Tensor: Image of shape `(..., Nx, Ny)`.
    """
    return ifftshift(ifft2(fftshift(k, dim=(-2,-1))), dim=(-2,-1)).real#.abs()


def get_cartesian_mask(shape, acceleration, mode='uniform'):
    size = shape[-2]
    n_keep = size / acceleration
    center_fraction = n_keep / 1000

    num_rows, num_cols = shape[-2], shape[-1]
    num_low_freqs = int(round(num_cols * center_fraction))

    # create the mask
    mask = torch.zeros((1, num_rows, num_cols))
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[..., pad: pad + num_low_freqs] = True
    # determine acceleration rate by adjusting for the number of low frequencies
    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
    offset = round(adjusted_accel) // 2

    accel_samples = torch.arange(offset, num_cols - 1, adjusted_accel)
    accel_samples = torch.round(accel_samples).int()
    mask[..., accel_samples] = True
    return mask


# def get_cartesian_mask(shape, n_keep=30):
#     # shape [Tuple]: (H, W)
#     size = shape[0]
#     center_fraction = n_keep / 1000
#     acceleration = size / n_keep

#     num_rows, num_cols = shape[0], shape[1]
#     num_low_freqs = int(round(num_cols * center_fraction))

#     # create the mask
#     mask = np.zeros((num_rows, num_cols), dtype=np.float32)
#     pad = (num_cols - num_low_freqs + 1) // 2
#     mask[:, pad: pad + num_low_freqs] = True

#     # determine acceleration rate by adjusting for the number of low frequencies
#     adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
#         num_low_freqs * acceleration - num_cols
#     )

#     offset = round(adjusted_accel) // 2

#     accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
#     accel_samples = np.around(accel_samples).astype(np.uint32)
#     mask[:, accel_samples] = True

#     return mask