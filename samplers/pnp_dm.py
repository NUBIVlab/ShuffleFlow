"""
    Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors.
    Adpated from https://github.com/devzhk/InverseBench/blob/main/algo/pnpdm.py.
    Official implementation: https://github.com/zihuiwu/PnP-DM-public.
"""

import torch
import tqdm

from . import BaseSampler
from .diffusion import ReverseSampler
from .langevin_dynamics import LangevinDynamics
from .scheduler import Scheduler


def get_exponential_decay_scheduler(num_steps, rho_max, rho_min, alpha=0.9):
    rho_steps = []
    rho = rho_max
    for i in range(num_steps):
        rho_steps.append(rho)
        rho = max(rho_min, rho * alpha)
    return rho_steps


class PnPDM(BaseSampler):
    def __init__(self, img_shape, net, forward_op, annealing_scheduler_config, diffusion_scheduler_config, lgvd_config, **kwargs):
        super(PnPDM, self).__init__(img_shape, net, forward_op)
        self.net.eval().requires_grad_(False)

        self.annealing_rhos = get_exponential_decay_scheduler(**annealing_scheduler_config)
        self.base_diffusion_scheduler = Scheduler(**diffusion_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    def __call__(self, observation, num_samples=1, verbose=True):

        num_steps = len(self.annealing_rhos)
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)

        x = torch.randn(num_samples, *self.img_shape, device=self.device)
        for step in pbar:
            rho = self.annealing_rhos[step]
            
            # 1. Langevin dynamics.
            z = self.lgvd.sample(x, self.forward_op, observation, rho, step / num_steps)

            # 2. Reverse diffusion.
            diffusion_scheduler = Scheduler.get_partial_scheduler(self.base_diffusion_scheduler, rho)
            sampler = ReverseSampler(diffusion_scheduler)
            with torch.no_grad():
                x = sampler.sample(self.net, z, SDE=True, verbose=False)
                
                if verbose:
                    loss = self.forward_op.loss(x, observation)
                    pbar.set_description(f'[PnP-DM] Avg. Error: {loss.sqrt().mean().cpu().item():.5e}')

        return x

