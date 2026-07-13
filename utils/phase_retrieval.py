import numpy as np
import torch
import torch.nn.functional as F
from piq import psnr
from torch.fft import fft2, fftshift, ifft2, ifftshift

# from utils.metrics import *


def get_pad(oversample: float, img_size: int=256) -> int:
    return int((oversample / 8.0) * img_size)


# def image_to_amplitude(x: torch.Tensor, oversample: float=4.0) -> torch.Tensor:
#     x = x.to(torch.float64)
#     pad = get_pad(oversample, x.shape[-1])
#     x_pad = F.pad(x, (pad, pad, pad, pad))
#     amplitude = ifftshift(fft2(fftshift(x_pad, dim=(-2,-1)), norm='ortho'), dim=(-2,-1)).abs()
#     return amplitude


# def amplitude_to_image(amplitude: torch.Tensor, crop: int=0) -> torch.Tensor:
#     img = ifftshift(ifft2(fftshift(amplitude, dim=(-2,-1)), norm='ortho'), dim=(-2,-1)).real
#     return img[..., crop:-crop, crop:-crop] if crop > 0 else img


def separate_modes(samples, gt1, gt2, psnr_min=5):
    samples1, samples2, samples3 = [], [], []

    for sample in samples:
        if torch.isnan(sample).any() or sample.abs().max() > 1.3:
            samples3.append(sample)
            continue
        psnr1 = psnr(sample.unsqueeze(0).clip(0, 1), gt1.clip(0, 1), data_range=1.0, reduction='none')
        psnr2 = psnr(sample.unsqueeze(0).clip(0, 1), gt2.clip(0, 1), data_range=1.0, reduction='none')
        if psnr1 > psnr2 and psnr1 > psnr_min:
            samples1.append(sample)
        elif psnr2 > psnr1 and psnr2 > psnr_min:
            samples2.append(sample)
        else:
            samples3.append(sample)
    # samples1, samples2, samples3 = \
    #     [np.ones([2, *gt1.shape])*np.nan if len(samples) == 0 else np.array(samples) for samples in [samples1, samples2, samples3]]
    samples1 = torch.stack(samples1) if len(samples1) > 0 else torch.zeros_like(gt1)
    samples2 = torch.stack(samples2) if len(samples2) > 0 else torch.zeros_like(gt1)
    samples3 = torch.stack(samples3) if len(samples3) > 0 else torch.zeros_like(gt1)
    return samples1, samples2, samples3