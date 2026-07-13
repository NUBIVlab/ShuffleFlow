import numpy as np
import torch
from torch.fft import fftn, ifftn


def get_forward_matrix(num_measurements, num_channels, img_size):
    np.random.seed(23)
    shape = (img_size, img_size)
    Areal = np.random.rand(num_measurements, *shape)
    Aimag = np.sqrt(1 - Areal ** 2)
    Areal = Areal * (-1) ** np.random.randint(10, size=Areal.shape)
    Aimag = Aimag * (-1) ** np.random.randint(10, size=Aimag.shape)
    return torch.from_numpy(Areal + 1j*Aimag)


def A_mult(A, x):
    # print(A.shape, x.shape)
    return fftn(A.to(x.device) * x, dim=(-2,-1), norm='ortho')


def A_tran(A, z):
    return A.conj().to(z.device) * ifftn(z, dim=(-2,-1), norm='ortho')