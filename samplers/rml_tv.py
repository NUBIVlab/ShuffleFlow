"""
    Regularized maximum likelihood (RML) with a Total Variation (TV) prior.
"""

import torch
import torch.nn as nn
import tqdm

from priors.tv import TotalVariation

from . import BaseSampler


class RML_TV(BaseSampler):
    def __init__(
        self,
        img_shape,
        tv_config,
        forward_op,
        num_steps=5000,
        lr=1e-2,
        **kwargs,
    ):
        super(RML_TV, self).__init__(img_shape, None, forward_op)
        self.tv_prior = TotalVariation(**tv_config)
        self.num_steps = num_steps
        self.lr = lr

    def __call__(self, observation, num_samples=1, verbose=True):
        x = nn.Parameter(torch.randn(num_samples, *self.img_shape, device=self.device))
        optimizer = torch.optim.Adam([x], lr=self.lr)

        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)
            likelihood = self.forward_op.loss(self.forward_op.unnormalize(x), observation).sum()
            prior = self.tv_prior(x)
            loss = likelihood + prior
            loss.backward()
            optimizer.step()

            if verbose:
                pbar.set_description(f'[RML-TV] Avg. Error: {likelihood.sqrt().mean().cpu().item():.5e}')

        return x.detach()
