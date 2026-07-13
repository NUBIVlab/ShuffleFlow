"""
    Annealed Langevin dynamics with a Total Variation prior (no learned score network).
    Based on the ALD structure from:
    Robust Compressed Sensing MRI with Deep Generative Priors, Instance-Optimal CS via Posterior Sampling.
    Adapted from https://github.com/devzhk/InverseBench/blob/main/algo/csgm_mri.py.
    Official implementation: https://github.com/utcsilab/csgm-mri-langevin.
"""

import numpy as np
import torch
import tqdm

from priors.tv import TotalVariation

from . import BaseSampler


def get_sigmas(sigmas_config):
    if sigmas_config.sigma_dist == 'geometric':
        sigmas = np.exp(np.linspace(np.log(sigmas_config.sigma_begin), np.log(sigmas_config.sigma_end), sigmas_config.num_steps))
    elif sigmas_config.sigma_dist == 'uniform':
        sigmas = np.linspace(sigmas_config.sigma_begin, sigmas_config.sigma_end, sigmas_config.num_steps)
    else:
        raise NotImplementedError('sigma distribution not supported')
    return sigmas


class ALD_TV(BaseSampler):
    """
    ALD with prior score -λ ∇ TV(x) (i.e. target density ∝ exp(-λ TV(x))).
    The diffusion ``net`` is unused; pass ``None`` if your config allows it.
    """

    def __init__(
        self,
        img_shape,
        tv_config,
        forward_op,
        sigmas_config,
        start_iter=1155,
        n_steps_each=3,
        step_lr=5e-5,
        mse=5,
        **kwargs,
    ):
        super(ALD_TV, self).__init__(img_shape, None, forward_op)
        self.sigmas_config = sigmas_config
        self.sigmas = get_sigmas(sigmas_config)
        self.start_iter = start_iter
        self.n_steps_each = n_steps_each
        self.step_lr = step_lr
        self.mse = mse
        self.tv_prior = TotalVariation(**tv_config)

    def tv_score(self, x):
        """Prior score ∇ log π(x) = -∇(weight * TV(x)) from ``priors.tv.TotalVariation``."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        tv = self.tv_prior(x)
        (grad_tv,) = torch.autograd.grad(tv, x, retain_graph=False, create_graph=False)
        return -grad_tv

    def __call__(self, observation, num_samples=1, verbose=True):
        pbar = tqdm.trange(self.start_iter, self.sigmas_config.num_steps) if verbose else range(self.start_iter, self.sigmas_config.num_steps)
        x_next = torch.randn(num_samples, *self.img_shape, device=self.device)

        for i in pbar:
            if i <= 1800:
                n_steps_each = 3
            else:
                n_steps_each = self.n_steps_each

            x_cur = x_next.detach().requires_grad_(True)
            sigma = self.sigmas[i]
            step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2
            step_size = torch.as_tensor(step_size, device=x_cur.device, dtype=x_cur.dtype)

            for _ in range(n_steps_each):
                x_cur = x_cur.detach().requires_grad_(True)
                meas_grad = self.forward_op.gradient(x_cur, observation, return_loss=False)
                x_det = x_cur.detach().requires_grad_(True)
                p_grad = self.tv_score(x_det)
                x_det = x_det.detach()
                meas_grad = meas_grad / torch.norm(meas_grad) * torch.norm(p_grad) * self.mse
                x_cur = x_det + step_size * (p_grad - meas_grad) + torch.sqrt(2 * step_size) * torch.randn_like(x_det)

            loss = self.forward_op.loss(x_cur, observation)
            if verbose:
                pbar.set_description(f'[ALD-TV] Avg. Error: {loss.sqrt().mean().cpu().item():.5e}')
            x_next = x_cur

        return x_next
