import numpy as np
import torch
import tqdm

from . import BaseSampler
from .scheduler import Scheduler


class ScoreMRI(BaseSampler):
    def __init__(self, img_shape, net, forward_op, scheduler_config, correct_steps=1, lr=0.16, lamb=1, **kwargs):
        super(ScoreMRI, self).__init__(img_shape, net, forward_op, **kwargs)
        self.scheduler = Scheduler(**scheduler_config)
        self.correct_steps = correct_steps
        self.lr = lr
        self.lamb = lamb

    def score(self, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = self.net(x, sigma)
        return (d - x) / sigma**2

    def project(self, x, observation):
        gradient = self.forward_op.gradient(x, observation, return_loss=False)
        return x - self.lamb * gradient

    def __call__(self, observation, num_samples=1, verbose=True):
        pbar = tqdm.trange(self.scheduler.num_steps)
        x_next = torch.randn(num_samples, *self.img_shape, device=self.device) * self.scheduler.sigma_max
        x_next.requires_grad = True

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]

            # Predictor step
            score = self.score(x_cur / self.scheduler.scaling_steps[i], sigma) / self.scheduler.scaling_steps[i]
            epsilon = torch.randn_like(x_cur)
            x_cur = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon

            # Projection
            x_cur = self.project(x_cur, observation)

            # Corrector steps
            for _ in range(self.correct_steps):
                sigma = self.scheduler.sigma_steps[i+1]

                score = self.score(x_cur / self.scheduler.scaling_steps[i+1], sigma) / self.scheduler.scaling_steps[i+1]
                epsilon = torch.randn_like(x_cur)
                lr = 2 * self.lr * torch.norm(epsilon)/torch.norm(score)
                x_cur = x_cur + lr * score + torch.sqrt(2*lr) * epsilon

                # Projection
                x_cur = self.project(x_cur, observation)
            x_next = x_cur
            
            if verbose:
                with torch.no_grad():
                    loss = self.forward_op.loss(x_cur, observation) # / np.prod(self.img_shape)
                pbar.set_description(f'[Score-MRI] Avg. Error: {loss.sqrt().mean().cpu().item():.5e}')
            
        return x_next