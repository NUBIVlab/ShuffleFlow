from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn import functional as F


def pad_double(img: torch.Tensor) -> torch.Tensor:
    H, W = img.shape[-2], img.shape[-1]
    return F.pad(img, (W//2, W//2, H//2, H//2))


def crop_half(img: torch.Tensor) -> torch.Tensor:
    H, W = img.shape[-2:]
    return img[...,H//4:3*H//4, W//4:3*W//4]


def embed_center(img: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Embed a smaller image in the center of a larger zero-padded array.
    Inverse of crop_half when target_shape is (2*H, 2*W) for img of shape (H, W).
    """
    *batch_dims, H, W = img.shape
    H_t, W_t = target_shape
    out = torch.zeros(*batch_dims, H_t, W_t, dtype=img.dtype, device=img.device)
    h_start, w_start = (H_t - H) // 2, (W_t - W) // 2
    out[..., h_start:h_start + H, w_start:w_start + W] = img
    return out


def pad_psf_to_shape(psf: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Pad PSF to target (H, W) shape, centered with zeros."""
    *batch_dims, H_psf, W_psf = psf.shape
    H_t, W_t = target_shape
    out = torch.zeros(*batch_dims, H_t, W_t, dtype=psf.dtype, device=psf.device)
    h_start = (H_t - H_psf) // 2
    w_start = (W_t - W_psf) // 2
    out[..., h_start:h_start + H_psf, w_start:w_start + W_psf] = psf
    return out


def get_mgrid(shape: tuple, range: tuple=(-1, 1)) -> torch.Tensor:
    """Generates a flattened grid of (x,y,...) coordinates in a given range.
    
    Args:
        shape (`tuple`): Shape of the datacude to be fitted.
        range (`tuple`, optional): Range of the grid. Defaults to `(-1, 1)`.

    Returns:
        `torch.Tensor`: Generated flattened grid of coordinates.
    """
    tensors = [torch.linspace(range[0], range[1], steps=N) if N > 1 else torch.tensor([0.0]) for N in shape]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid.reshape(-1, len(shape))


def get_total_params(model: torch.nn.Module) -> int:
    """Calculate the total number of parameters in a model.
	
	Args:
		model (`torch.nn.Module`): PyTorch module.

	Returns:
		`int`: Total number of parameters in the model.
	"""
    return sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])


def get_fourier_coord(N: int=80, l: float=3.2e-3, device: str='cuda:0') -> tuple:
	"""Calculate the 2D Fourier space polar coordinates for a given grid size.

	Args:
		N (`int`, optional): Grid size [pixels]. Defaults to `80`.
		l (`float`, optional): Grid length [m]. Defaults to `3.2e-3`.
		device (`str`, optional): Device of the coordinate tensors. Defaults to `'cuda:0'`.

	Returns:
		`tuple`: 2D Fourier space polar coordinates `[k2D, theta2D]`.
	"""
	fx1D = torch.linspace(-np.pi/l, np.pi/l, N, requires_grad=False, device=device)
	fy1D = torch.linspace(-np.pi/l, np.pi/l, N, requires_grad=False, device=device)
	[fx2D, fy2D] = torch.meshgrid(fx1D, fy1D, indexing='xy')
	k2D = torch.sqrt(fx2D**2 + fy2D**2) * N
	theta2D = torch.arctan2(fy2D, fx2D) + np.pi/2 # Add `np.pi/2` to match the polar definition of the theta.
	return k2D, theta2D % (2*torch.pi)


def get_conv_matrix(image_size, kernel):
    K = kernel.shape[0]
    H = torch.zeros(((image_size+K-1)**2, image_size**2))
    for i in range(image_size):
        for j in range(image_size):
            zeros = torch.zeros((image_size+K-1, image_size+K-1))
            zeros[i:i+K, j:j+K] = kernel
            H[:,i*image_size+j] = zeros.view(-1)
            
    mask = torch.zeros((image_size+K-1, image_size+K-1))
    mask[K//2:image_size+K//2, K//2:image_size+K//2] = 1
    H = H[mask.view(-1)==1]
    return H


def gaussian_posterior(y: torch.Tensor, F: torch.Tensor, Sigma: torch.Tensor, mu: torch.Tensor, Gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Calculate the mean and covariance of a Gaussian posterior distribution.

	Args:
		y (`torch.Tensor`): Measurement.
		F (`torch.Tensor`): Forward operator.
		Sigma (`torch.Tensor`): Covariance matrix of Gaussian likelihood.
		mu (`torch.Tensor`): Mean of Gaussian prior.
		Gamma (`torch.Tensor`): Covariance matrix of Gaussian prior.

	Returns:
		tuple[`torch.Tensor`, `torch.Tensor`]: Mean and covariance matrix of the posterior.
	"""
	m = mu + Gamma @ F.T @ torch.inverse(Sigma + F @ Gamma @ F.T) @ (y - F @ mu)
	C = Gamma - Gamma @ F.T @ torch.inverse(Sigma + F @ Gamma @ F.T) @ F @ Gamma
	return m, C


def batched_sum(x: torch.Tensor) -> torch.Tensor:
	"""Sums all dimensions of a tensor except the batch dimension.
    
	Args:
		x (`torch.Tensor`): Input tensor.
	
	Returns:
		`torch.Tensor`: Tensor with summed dimensions.
	"""
	if len(x.shape) <= 1:
		return x
	return x.sum(dim=tuple(range(1, len(x.shape))))


def convolve(img: torch.Tensor, psf: torch.Tensor, mode: str='fft', pad: str='none') -> torch.Tensor:
    """2D convolution of two images.

    Args:
        img (`torch.Tensor`): Input image.
        psf (`torch.Tensor`): Point spread function.

    Returns:
        `Tensor`: Convolved image.
    """
    if mode == 'fft':
        if pad == 'double':
            img = pad_double(img)
            psf = pad_double(psf)
            return crop_half(fftshift(ifft2(fft2(ifftshift(img, dim=(-2,-1))) * fft2(ifftshift(psf, dim=(-2,-1)))), dim=(-2,-1)).real)
        else:
            return fftshift(ifft2(fft2(ifftshift(img, dim=(-2,-1))) * fft2(ifftshift(psf, dim=(-2,-1)))), dim=(-2,-1)).real
    elif mode == 'conv':
        img = F.pad(img.unsqueeze(0), ((psf.shape[-2] - 1) // 2, (psf.shape[-2] - 1) // 2, (psf.shape[-1] - 1) // 2, (psf.shape[-1] - 1) // 2), mode=pad)
        return F.conv2d(img, psf.unsqueeze(1), groups=3).squeeze(0)
    else:
        raise ValueError(f'Invalid convolution mode: {mode}')


def get_gaussian_kernel(shape: tuple, sigma: float) -> torch.Tensor:
    """Generate a 2D Gaussian kernel.

    Args:
        size (`int`): Kernel size.
        sigma (`float`): Standard deviation.

    Returns:
        `torch.Tensor`: Gaussian kernel.
    """
    x = torch.linspace(-shape[0]//2+1/2, shape[0]//2-1/2, shape[0])
    y = torch.linspace(-shape[1]//2+1/2, shape[1]//2-1/2, shape[1])
    X, Y = torch.meshgrid(x, y, indexing='xy')
    kernel = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, shape[0], shape[1])
    return kernel


def get_global_average(local_value: torch.Tensor, world_size: int) -> torch.Tensor:
    """Computes the average value across all processes (GPUs).

    Args:
        local_value (`torch.Tensor`): Local value to be averaged.
        world_size (`int`): Number of processes (GPUs).

    Returns:
        `float`: Global average value.
    """
    # 1. Clone the loss and detach it from the graph so we don't affect gradients
    #    It must be a Tensor on the correct device.
    reduced_value = local_value.detach().clone()
    
    # 2. Sum the losses across all processes
    dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
    
    # 3. Average the sum by dividing by the number of GPUs
    global_average_value = reduced_value / world_size
    
    return global_average_value.item()


def gather_tensors(local_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Collects a tensor from every GPU and stacks them.
    
    Args:
        local_tensor (`torch.Tensor`): Tensor on current GPU.
        
    Returns:
        `list[torch.Tensor]`: List of tensors from all GPUs.
    """
    # 1. Prepare a list to hold the results
    # We must allocate tensors of the same size on the destination device
    world_size = dist.get_world_size()
    local_tensor = local_tensor.contiguous()
    gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    
    # 2. Collect (This blocks until all processes arrive)
    dist.all_gather(gathered_list, local_tensor)
    
    # 3. (Optional) Stack them into one big tensor
    # If local_tensor was (32, 10), output is now (World_Size * 32, 10)
    return torch.cat(gathered_list, dim=0)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if param.requires_grad:
            ema_name = name.replace('module.', '')
            ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)


def unwrap_model(model):
    """
    Unwrap a model from any distributed or compiled wrappers. 
    """
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        model = model._orig_mod
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model
