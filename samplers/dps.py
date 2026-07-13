"""
    Diffusion Posterior Sampling for General Noisy Inverse Problems.
    Adpated from https://github.com/devzhk/InverseBench/blob/main/algo/dps.py.
    Official implementation: https://github.com/DPS2022/diffusion-posterior-sampling.
"""

import numpy as np
import torch
import tqdm

from . import BaseSampler
from .scheduler import Scheduler


class DPS(BaseSampler):
    def __init__(self, img_shape, net, forward_op, diffusion_scheduler_config, zeta=1.0, sde=True, **kwargs):
        super(DPS, self).__init__(img_shape, net, forward_op, **kwargs)
        self.zeta = zeta
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        
    def __call__(self, observation, num_samples=1, verbose=True, **kwargs):

        x_next = torch.randn(num_samples, *self.img_shape, device=self.device) * self.scheduler.sigma_max
        x_next.requires_grad = True

        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)
        
        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
            
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
            gradient, loss = self.forward_op.gradient(denoised, observation, return_loss=True)
            
            llh_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            llh_grad = llh_grad * 0.5 / loss.sqrt().view(-1, 1, 1, 1)

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5 
            x_next -= llh_grad * self.zeta
            
            if verbose:
                pbar.set_description(f'[DPS] Avg. Error: {loss.sqrt().mean().cpu().item():.5e}')
                
        return x_next