"""
    Robust Compressed Sensing MRI with Deep Generative Priors, Instance-Optimal Compressed Sensing via Posterior Sampling.
    Adpated from https://github.com/devzhk/InverseBench/blob/main/algo/csgm_mri.py.
    Official implementation: https://github.com/utcsilab/csgm-mri-langevin.
"""

import numpy as np
import torch
import tqdm

from . import BaseSampler


def get_sigmas(sigmas_config):
    if sigmas_config.sigma_dist == 'geometric':
        sigmas = np.exp(np.linspace(np.log(sigmas_config.sigma_begin), np.log(sigmas_config.sigma_end), sigmas_config.num_steps))
    elif sigmas_config.sigma_dist == 'uniform':
        sigmas = np.linspace(sigmas_config.sigma_begin, sigmas_config.sigma_end, sigmas_config.num_steps)
    else:
        raise NotImplementedError('sigma distribution not supported')
    return sigmas


class ScoreALD(BaseSampler):
    def __init__(self, img_shape, net, forward_op, sigmas_config, start_iter=1155, n_steps_each=3, step_lr=5e-5, mse=5, **kwargs):
        super(ScoreALD, self).__init__(img_shape, net, forward_op, **kwargs)
        self.sigmas_config = sigmas_config
        self.sigmas = get_sigmas(sigmas_config)
        self.start_iter = start_iter
        self.n_steps_each = n_steps_each
        self.step_lr = step_lr
        self.mse = mse

    def score(self, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = self.net(x, sigma)
        return (d - x) / sigma**2

    def __call__(self, observation, num_samples=1, verbose=True):
        pbar = tqdm.trange(self.start_iter, self.sigmas_config.num_steps) if verbose else range(self.start_iter, self.sigmas_config.num_steps)
        x_next = torch.randn(num_samples, *self.img_shape, device=self.device)
        # x_next.requires_grad = True

        for i in pbar:
            if i <= 1800:
                n_steps_each = 3
            else:
                n_steps_each = self.n_steps_each

            x_cur = x_next.detach().requires_grad_(True)
            sigma = self.sigmas[i]
            step_size = torch.tensor(self.step_lr * (sigma / self.sigmas[-1]) ** 2)

            for _ in range(n_steps_each):
                meas_grad = self.forward_op.gradient(x_cur, observation, return_loss=False)
                x_cur = x_cur.detach()
                with torch.no_grad():
                    p_grad = self.score(x_cur, sigma)
                    meas_grad /= torch.norm(meas_grad)
                    meas_grad *= torch.norm(p_grad)
                    meas_grad *= self.mse
                    x_cur = x_cur + step_size * (p_grad - meas_grad) + torch.sqrt(2*step_size) * torch.randn_like(x_cur)
            x_next = x_cur
            
            if verbose:
                with torch.no_grad():
                    loss = self.forward_op.loss(x_cur, observation) # / np.prod(self.img_shape)
                pbar.set_description(f'[Score-ALD] Avg. Error: {loss.sqrt().mean().cpu().item():.5e}')
            
        return x_next