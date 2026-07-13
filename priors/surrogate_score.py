import torch
from torch import nn

from utils.sde import *


class SurrogateScore(nn.Module):
    def __init__(self, weight, sde_config, model, **kwargs):
        super().__init__()
        self.weight = weight
        self._sde = get_sde(**sde_config)
        self.model = model
        # self.Nt = config.flow.loss.Nt
        # self.t_decay_rate = config.flow.loss.t_decay_rate
        # self.t_decay_min = config.flow.loss.t_decay_min
        self._score_fn = get_score_fn(sde=self._sde, model=self.model, continuous=True)
        
    def forward(self, x, epoch=None):
        # self._score_fn = get_score_fn(sde=self._sde, model=self.model, continuous=True)
        # x = x.repeat(1,self.Nt,1,1).view(x.shape[0]*self.Nt,1,x.shape[-2],x.shape[-1])
        
        # t_max = max(1 - self.t_decay_rate*epoch, self.t_decay_min)
        t = torch.rand(x.shape[0], device=x.device) #* t_max
        
        mean, std = self._sde.marginal_prob(x, t.view(-1,1,1,1))
        z = torch.randn_like(x)
        x_t = mean + std * z
        _, g = self._sde.sde(x_t, t.view(-1,1,1,1)) 
        beta_t = g ** 2

        score = self._score_fn(x_t, t)
        grad = - z / std
        score_error = ((score - grad)**2).sum(dim=(-3,-2,-1), keepdims=True)
        grad_norm = (grad ** 2).sum(dim=(-3,-2,-1), keepdims=True)
        drift_div = - 0.5 * beta_t * x.shape[-1] * x.shape[-2]
        integral = (g ** 2 * (score_error - grad_norm) - 2 * drift_div)
        
        mean, std = self._sde.marginal_prob(x, torch.ones((x.shape[0],1,1,1), device=x.device) * self._sde.T)
        prior_logp = self._sde.prior_logp(mean + std * torch.randn_like(x)).view(-1,1,1,1)
        logp_bound = prior_logp - 0.5 * integral
        # logp_bound = logp_bound.view(-1, self.Nt).mean(-1)
        return - self.weight * logp_bound.squeeze()
        