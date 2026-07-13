"""Forward operator for single-coil compressed sensing MRI."""

import numpy as np
import torch

from . import BaseOperator


class SingleCoilMRI(BaseOperator):
    def __init__(self, mask: torch.Tensor=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask = mask
    
    @staticmethod
    def ifft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def fft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x
    
    @staticmethod
    def get_cartesian_mask(total_lines: int=320, acceleration: float=4.0, orientation: str='vertical', pattern: str='random', seed: int=0) -> np.ndarray:
        np.random.seed(seed)
        
        if 1 < acceleration <= 6:
            # Keep 8% of center samples.
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples.
            acs_lines = np.floor(0.04 * total_lines).astype(int)
        
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / acceleration)

        # Get locations of ACS lines (!!! Assumes k-space is even sized and centered, true for fastMRI).
        center_line_idx = np.arange((total_lines - acs_lines) // 2, (total_lines + acs_lines) // 2)
        
        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx, size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines (!!! Only supports integer for now).
            random_line_idx = outer_line_idx[::int(acceleration)]
        else:
            raise NotImplementedError(f'Mask pattern {pattern} not implemented.')

        # Create a mask and place ones at the right locations.
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        mask = torch.from_numpy(mask)
        
        if orientation == 'vertical':
            mask = mask[None, None, :].bool().repeat(1, total_lines, 1)
        elif orientation == 'horizontal':
            mask = mask[None, :, None].bool().repeat(1, 1, total_lines)
        else:
            raise NotImplementedError

        return mask
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(self.mask * self.fft(img))

    def transpose(self, kspace: torch.Tensor) -> torch.Tensor:
        return self.ifft(torch.view_as_complex(kspace))
    
    def load_parameters(self, data: dict):
        self.mask = data['mask'].to(self.device)
    