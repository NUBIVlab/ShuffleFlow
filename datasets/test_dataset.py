import os
from typing import List

import numpy as np
import torch

from forward_operators.single_coil_mri import SingleCoilMRI
from utils.name import *
from utils.torch import *

from . import BaseDataset


class SingleCoilMRIDataset(BaseDataset):
    def __init__(self, data_path: str, group: str, name_data: str, name_operator: str, idx_list: List[int]=None, **kwargs):
        super().__init__(name_data, name_operator, idx_list, **kwargs)
        self.dataset_path = os.path.join(data_path, group, name_data)
        self.gt_path = os.path.join(self.dataset_path, 'gt_val')
        self.mask_path = os.path.join(self.dataset_path, f'mask_{self.name_operator}')
        self.kspace_path = os.path.join(self.dataset_path, f'kspace_{self.name_operator}')

    def __getitem__(self, idx):
        data_idx = self.idx_map(idx)
        gt = torch.from_numpy(np.load(os.path.join(self.gt_path, f'gt_{data_idx}.npy')))
        mask = torch.from_numpy(np.load(os.path.join(self.mask_path, f'mask_{data_idx}.npy')))
        kspace = torch.from_numpy(np.load(os.path.join(self.kspace_path, f'kspace_{data_idx}.npy')))
        zero_filled = SingleCoilMRI(mask).transpose(kspace).real
        return {'gt': gt, 'measurement': kspace, 'mask': mask, 'zero_filled': zero_filled}


class DeblurDataset(BaseDataset):
    def __init__(self, data_path: str, group: str, name_data: str, name_operator: str, idx_list: List[int]=None, **kwargs):
        super().__init__(name_data, name_operator, idx_list, **kwargs)
        self.dataset_path = os.path.join(data_path, 'test', name_data)
        self.gt_path = os.path.join(self.dataset_path, 'gt')
        self.psf_path = os.path.join(self.dataset_path, f'psf_{self.name_operator}')
        self.obs_path = os.path.join(self.dataset_path, f'obs_{self.name_operator}')

    def __getitem__(self, idx):
        data_idx = self.idx_map(idx)
        gt = torch.from_numpy(np.load(os.path.join(self.gt_path, f'gt_{data_idx}.npy')))
        psf = torch.from_numpy(np.load(os.path.join(self.psf_path, f'psf_{data_idx}.npy')))
        obs = torch.from_numpy(np.load(os.path.join(self.obs_path, f'obs_{data_idx}.npy')))
        return {'gt': gt, 'measurement': obs, 'psf': psf}


class PhaseRetrievalDataset(BaseDataset):
    def __init__(self, data_path: str, group: str, name_data: str, name_operator: str, idx_list: List[int]=None, **kwargs):
        super().__init__(name_data, name_operator, idx_list, **kwargs)
        self.dataset_path = os.path.join(data_path, 'test', name_data)
        self.gt_path = os.path.join(self.dataset_path, 'gt')
        self.amp_path = os.path.join(self.dataset_path, f'amp_{self.name_operator}')

    def __getitem__(self, idx):
        data_idx = self.idx_map(idx)
        gt = torch.from_numpy(np.load(os.path.join(self.gt_path, f'gt_{data_idx}.npy'))).float()
        amp = torch.from_numpy(np.load(os.path.join(self.amp_path, f'amp_{data_idx}.npy')))
        return {'gt': gt, 'measurement': amp}
