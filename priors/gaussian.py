import torch
from torch import nn
from torch.distributions import MultivariateNormal


class Gaussian(nn.Module):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, **kwargs):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.mvn = MultivariateNormal(loc=mean.flatten(), covariance_matrix=cov)
    
        
    def forward(self, x, **kwargs):
        return - self.mvn.log_prob(x.view(x.shape[0], -1))
    
    

# class GaussianPrecond(nn.Module):
#     def __init__(self, mean: torch.Tensor, cov: torch.Tensor, **kwargs):
#         super().__init__()
#         self.mean = mean
#         self.cov = cov
    
#     def score(self, x):
#         return - self.cov.inverse() @ (x - self.mean)
    
#     def forward(self, x):
        
