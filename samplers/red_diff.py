"""
    RED-Diff: A Variational Perspective on Solving Inverse Problems with Diffusion Models.
    Adpated from https://github.com/devzhk/InverseBench/blob/main/algo/reddiff.py.
    Official implementation: https://github.com/NVlabs/RED-diff.
"""


import numpy as np
import torch
import tqdm

from . import BaseSampler
from .scheduler import Scheduler


class REDDiff(BaseSampler):
    def __init__(self, img_shape, net, forward_op, num_steps=1000, measurement_weight=1.0, base_lambda=0.25, base_lr=0.5, lambda_schedule='constant', **kwargs):
        super(REDDiff, self).__init__(img_shape, net, forward_op)
        self.net.eval().requires_grad_(False)

        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp', timestep='vp', scaling='vp')
        self.base_lr = base_lr
        self.measurement_weight = measurement_weight
        if lambda_schedule == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_schedule == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_schedule == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def __call__(self, observation, num_samples=1, verbose=True):
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)

        # 0. Random initialization (instead of from pseudo-inverse).
        mu = torch.zeros(num_samples, self.img_shape[0], self.img_shape[1], self.img_shape[2], device=self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))
        for step in pbar:
            # 1. Forward diffusion.
            with torch.no_grad():
                sigma, scaling = self.scheduler.sigma_steps[step], self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            # 2. Regularized optimization.
            lam = self.lambda_fn(sigma)  # sigma here equals to 1/SNR
            optimizer.zero_grad()

            gradient, loss = self.forward_op.gradient(mu, observation, return_loss=True)
            gradient = gradient * self.measurement_weight + lam * (pred_epsilon - epsilon)
            mu.grad = gradient

            optimizer.step()
            if verbose:
                pbar.set_description(f'[RED-Diff] Avg. Error: {torch.sqrt(loss).mean().cpu().item()/num_samples:.5e}')
            # if wandb.run is not None:
            #     wandb.log({'data_fitting_loss': torch.sqrt(loss)}, step=step)
        return mu

