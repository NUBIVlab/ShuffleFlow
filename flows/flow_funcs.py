import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from . import BaseFlow


class AffineFlow(BaseFlow):
    def __init__(self):
        super().__init__()
        
    def forward(self, z, params):
        s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(s)
        s = torch.exp(log_s)

        # Compute flow.
        z = s * (z + t)
        
        # Compute log|detJi|.
        log_det_Ji = log_s
        
        return z, log_det_Ji

    def reverse(self, z, params):
        s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(s)
        s = torch.exp(log_s)
        
        # Compute flow.
        z = z / s - t
        
        # Compute log|detJi|.
        log_det_Ji = - log_s
        
        return z, log_det_Ji
    
    def sample(self, z, params):
        s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(s)
        s = torch.exp(log_s)

        # Compute flow.
        z = z / s - t
        
        return z


class PlanarFlow(BaseFlow):
    def __init__(self):
        super().__init__()
        self.h = nn.Tanh()
        self.softplus = nn.Softplus()
    
    def h_derivative(self, x):
        return 1 - self.h(x) ** 2
    
    def forward(self, z, params):
        u, w, b = params.chunk(3, dim=-1)
        w = torch.exp(2*torch.tanh(w))

        # Reparameterize u for invertiblility.
        u = (-1 + self.softplus(u * w)) / w

        # Compute flow.
        wzb = w * z + b
        z = z + u * self.h(wzb)

        # Compute log|detJi|.
        psi = w * self.h_derivative(wzb)
        log_det_Ji = ((1 + u * psi).abs()).log()

        return z, log_det_Ji
    
    def sample(self, z, params):
        u, w, b = params.chunk(3, dim=-1)
        w = torch.exp(2*torch.tanh(w))
        
        # Reparameterize u for invertiblility.
        u = (-1 + self.softplus(u * w)) / w

        # Compute flow.
        wzb = w * z + b
        z = z + u * self.h(wzb)
        
        return z
    
    
class RadialFlow(BaseFlow):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        
    def h(self, alpha, r):
        return 1 / (alpha + r)
    
    def h_derivative(self, h):
        return - h ** 2
    
    def forward(self, z, params):
        z0, alpha, beta = params.chunk(3, dim=-1)
        
        # Reparameterize alpha and beta for invertiblility.
        alpha = self.softplus(alpha)
        beta = - alpha + self.softplus(beta)
        
        # Compute flow.
        r = torch.abs(z-z0)
        h = self.h(alpha, r)
        z = z + beta * h * (z - z0)
        
        # Compute log|detJi|.
        log_det_Ji = (1 + beta * h + beta * r * self.h_derivative(h)).abs().log()
        
        return z, log_det_Ji

    def sample(self, z, params):
        z0, alpha, beta = params.chunk(3, dim=-1)
        
        # Reparameterize alpha and beta for invertiblility.
        alpha = self.softplus(alpha)
        beta = - alpha + self.softplus(beta)
        
        # Compute flow.
        r = torch.abs(z-z0)
        h = self.h(alpha, r)
        z = z + beta * h * (z - z0)
        
        return z

# class SplineFlow(BaseFlow):
#     def __init__(self):
#         super(SplineFlow, self).__init__()
        
#     def forward(self, z, params):
#         widths, heights, deltas = params.chunk(3, dim=-1)
        
#         z, log_det = self.spline_transform(z, widths, heights, deltas)
#         return z, log_det
    
#     def spline_transform(self, x, widths, heights, deltas):
#         """Performs a simple rational-quadratic spline transform."""
#         cdf = torch.cumsum(widths, dim=-1)
#         cdf = F.pad(cdf, (1, 0), value=0.0)
#         bin_indices = torch.searchsorted(cdf, torch.rand_like(x))
        
#         # Get bin parameters
#         x_min = cdf[..., :-1]
#         x_max = cdf[..., 1:]
#         y_min = torch.cumsum(heights, dim=-1)[..., :-1]
#         y_max = torch.cumsum(heights, dim=-1)[..., 1:]
#         delta = deltas[..., bin_indices - 1]
        
#         # Linear interpolation (simplified for demonstration)
        
        
#         return z, log_det


class SplineFlow(nn.Module):
    def __init__(self, num_bins=16, bound=3.0):
        super().__init__()
        self.num_bins = num_bins
        # self.bound = bound
        self.softplus = nn.Softplus()
        
    def forward(self, x, params, inverse=False):
        """
        Args:
            inputs: Tensor of shape [..., 1]
            params: Tensor of shape [..., 3*num_bins-1]
            inverse: Whether to compute inverse transformation
        """
        unnormalized_widths, unnormalized_heights = params.chunk(2, dim=-1)

        outputs, logabsdet = self.quadratic_spline(
            inputs=x,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse
        )
        return outputs, logabsdet

    def forward(self, z, params, inverse=True):
        """
        Args:
            inputs: Tensor of shape [..., 1]
            params: Tensor of shape [..., 3*num_bins-1]
            inverse: Whether to compute inverse transformation
        """
        # unnormalized_widths = params[..., :self.num_bins]
        # unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_widths, unnormalized_heights = params.chunk(2, dim=-1)
        
        outputs, logabsdet = self.quadratic_spline(
            inputs=z,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse
        )
        return outputs, logabsdet
    
    @staticmethod
    def quadratic_spline(inputs,
                        unnormalized_widths,
                        unnormalized_heights,
                        inverse=False,
                        left=0., right=1., bottom=0., top=1.,
                        min_bin_width=1e-3,
                        min_bin_height=1e-3):
        
        if inverse:
            inputs = (inputs - bottom) / (top - bottom)
        else:
            inputs = (inputs - left) / (right - left)

        num_bins = unnormalized_widths.shape[-1]

        if min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        widths = F.softmax(unnormalized_widths, dim=-1)
        unnorm_heights_exp = torch.exp(unnormalized_heights)
        
        if unnorm_heights_exp.shape[-1] == num_bins - 1:
            # Set boundary heights s.t. after normalization they are exactly 1.
            first_widths = 0.5 * widths[..., 0]
            last_widths = 0.5 * widths[..., -1]
            numerator = (0.5 * first_widths * unnorm_heights_exp[..., 0]
                        + 0.5 * last_widths * unnorm_heights_exp[..., -1]
                        + torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                    * widths[..., 1:-1], dim=-1))
            constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
            constant = constant[..., None]
            unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)
        # print(unnorm_heights_exp[..., :-1].shape, unnorm_heights_exp[..., 1:].shape)
        unnormalized_area = torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths, dim=-1)[..., None]
        
        heights = unnorm_heights_exp / unnormalized_area
        heights = min_bin_height + (1 - min_bin_height) * heights
        # print(heights.shape, unnormalized_heights.shape)
        bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
        bin_left_cdf[..., -1] = 1.
        bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode='constant', value=0.0)

        bin_locations = torch.cumsum(widths, dim=-1)
        bin_locations[..., -1] = 1.
        bin_locations = F.pad(bin_locations, pad=(1, 0), mode='constant', value=0.0)
        
        bin_locations[..., -1] += 1+1e-6
        if inverse:
            bin_idx = torch.clamp(torch.searchsorted(bin_left_cdf, inputs, right=True) - 1, 0, bin_left_cdf.size(-1) - 2)# torch.searchsorted(bin_left_cdf, inputs)[..., None]
        else:
            bin_idx = torch.clamp(torch.searchsorted(bin_locations, inputs, right=True) - 1, 0, bin_locations.size(-1) - 2) # torch.searchsorted(bin_locations, inputs)[..., None]
            
        # print(bin_idx.shape)
        
        input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
        
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
        # print(input_bin_widths)

        input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]

        input_left_heights = heights.gather(-1, bin_idx)[..., 0]
        input_right_heights = heights.gather(-1, bin_idx+1)[..., 0]

        a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
        b = input_left_heights * input_bin_widths
        c = input_left_cdf

        if inverse:
            # print(c.shape, inputs.shape)
            inputs = inputs.squeeze(-1)
            c_ = c - inputs
            alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c_)) / (2*a)
            outputs = alpha * input_bin_widths + input_bin_locations
            outputs = torch.clamp(outputs, 0, 1)
            logabsdet = -torch.log((alpha * (input_right_heights - input_left_heights)
                                    + input_left_heights))
        else:
            alpha = (inputs - input_bin_locations) / input_bin_widths
            outputs = a * alpha.pow(2) + b * alpha + c
            outputs = torch.clamp(outputs, 0, 1)
            logabsdet = torch.log((alpha * (input_right_heights - input_left_heights)
                                + input_left_heights))

        if inverse:
            outputs = outputs * (right - left) + left
            logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
        else:
            outputs = outputs * (top - bottom) + bottom
            logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)

        # return outputs, logabsdet
        return outputs.unsqueeze(-1), logabsdet.unsqueeze(-1)