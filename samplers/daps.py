"""
    Improving diffusion inverse problem solving with decoupled noise annealing.
    Adpated from https://github.com/devzhk/InverseBench/blob/main/algo/daps.py.
    Official implementation: https://github.com/zhangbingliang2019/DAPS.
"""

import numpy as np
import torch
import tqdm

from . import BaseSampler
from .diffusion import ReverseSampler
from .langevin_dynamics import LangevinDynamics
from .scheduler import Scheduler


class DAPS(BaseSampler):
    def __init__(self, img_shape, net, forward_op, annealing_scheduler_config, diffusion_scheduler_config, lgvd_config, **kwargs):
        super(DAPS, self).__init__(img_shape, net, forward_op, **kwargs)
        self.net.eval().requires_grad_(False)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    
    def __call__(self, observation, num_samples=1, verbose=True):
        """Samples using the DAPS method.

            Parameters:
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
        # if num_samples > 1:
        #     observation = observation.repeat(num_samples, 1, 1, 1)
        # device = self.forward_op.device
        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)
        xt = torch.randn(num_samples, *self.img_shape, device=self.device) * self.annealing_scheduler.sigma_max
        
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. Reverse diffusion.
            diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = ReverseSampler(diffusion_scheduler)
            with torch.no_grad():
                x0hat = sampler.sample(self.net, xt, SDE=False, verbose=False)

            # 2. Langevin dynamics.
            x0y = self.lgvd.sample(x0hat, self.forward_op, observation, sigma, step / self.annealing_scheduler.num_steps)
            del x0hat  # release ref so diffusion output can be freed

            # 3. Forward diffusion.
            with torch.no_grad():
                # if step < self.annealing_scheduler.num_steps - 1:
                xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
                # else:
                #     xt = x0y
            del x0y

            if verbose:
                loss = self.forward_op.loss(self.forward_op.unnormalize(xt), observation).sqrt().mean()
                pbar.set_description(f'[DAPS] Avg. Error: {loss.sqrt().mean().cpu().item():.5e}')
        del diffusion_scheduler, sampler
        return xt

