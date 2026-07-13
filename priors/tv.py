from torch import nn


class TotalVariation(nn.Module):
    def __init__(self, weight: float=1.0, mode: str='anisotropic', **kwargs):
        super().__init__()
        self.weight = weight
        self.mode = mode
        
    def forward(self, x, **kwargs):
        if self.mode == 'anisotropic':
            return self.weight * ((x[...,:, :-1] - x[...,:, 1:]).abs().mean(0).sum() + (x[...,:-1, :] - x[...,1:, :]).abs().mean(0).sum())
        elif self.mode == 'isotropic':
            return self.weight * ((x[...,:-1, :-1] - x[...,:-1, 1:])**2 + (x[...,:-1, :-1] - x[...,1:, :-1])**2).sqrt().mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
