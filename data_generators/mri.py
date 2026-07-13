"""Data generator for single-coil compressed sensing MRI."""

import json
import os

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from forward_operators import SingleCoilMRI
from utils.visualization import visualize_data_mri

from . import BaseDataGenerator


class MRIDataGenerator(BaseDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data_path = os.path.join(self.data_config.data_path, self.data_config.group)
        self.dataset_path = os.path.join(self.data_path, self.name_data)

        # Create directories.
        os.makedirs(os.path.join(self.data_path, self.name_data, 'gt_train'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, self.name_data, 'gt_val'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, self.name_data, f'mask_{self.name_operator}'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, self.name_data, f'kspace_{self.name_operator}'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, self.name_data, f'vis_{self.name_operator}'), exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.data_config.img_size, self.data_config.img_size)),
        ])
        
        # # Save dataset information.
        # info_file = os.path.join(self.data_path, self.name_data, f'vis_{self.name_operator}', 'info.json')
        # info = {'config': OmegaConf.to_container(self.inverse_problem_config, resolve=True), }
        # with open(info_file, 'w') as f:
        #     json.dump(info, f)
        # self.logger.info('Dataset information saved to %s.', info_file)
        
    def generate_data(self):
        self.logger.info('Start generating data for %s using %s dataset.', self.operator_config.name, self.data_config.name)

        for split in ['val', 'train']:
            files = os.listdir(os.path.join(self.data_path, f'{self.data_config.coil}coil_{split}'))
            files.sort()
              
            # Create directories.
            if split == 'val':
                os.makedirs(os.path.join(self.dataset_path, f'mask_{self.name_operator}'), exist_ok=True)
                os.makedirs(os.path.join(self.dataset_path, f'kspace_{self.name_operator}'), exist_ok=True)
                os.makedirs(os.path.join(self.dataset_path, f'vis_{self.name_operator}'), exist_ok=True)
            
            sample_idx = 0
            for file in tqdm(files, desc=f'Generating {split} data', total=len(files)):
                data = h5py.File(os.path.join(self.data_path, f'{self.data_config.coil}coil_{split}', file), 'r')
                rec_rss = np.array(data['reconstruction_rss'])
                if split == 'train':
                    for gt in rec_rss[10:34]:
                        gt = self.transform(gt).numpy()
                        if self.data_config.normalize:
                            gt = (gt - gt.min()) / (gt.max() - gt.min())
                        np.save(os.path.join(self.dataset_path, f'gt_{split}', f'gt_{sample_idx}.npy'), gt)
                        sample_idx += 1
                else:
                    gt = self.transform(rec_rss[rec_rss.shape[0]//2,...])
                    if self.data_config.normalize:
                        gt = (gt - gt.min()) / (gt.max() - gt.min())
                    if self.operator_config.mask.name == 'cartesian':
                        mask = SingleCoilMRI.get_cartesian_mask(
                            total_lines=gt.shape[-2],
                            acceleration=self.operator_config.mask.acceleration,
                            orientation=self.operator_config.mask.orientation,
                            pattern=self.operator_config.mask.pattern,
                            seed=self.operator_config.mask.seed
                        )
                    else:
                        raise NotImplementedError(f'Unsupported mask: {self.operator_config.mask.name}.')
                    
                    kspace = SingleCoilMRI(mask=mask, sigma_noise=self.operator_config.sigma_noise)(gt)
                    kspace_full = torch.view_as_real(SingleCoilMRI.fft(gt))
                    zero_filled = SingleCoilMRI(mask).transpose(kspace).real
                    
                    gt, mask, kspace, kspace_full, zero_filled = gt.numpy(), mask.numpy(), kspace.numpy(), kspace_full.numpy(), zero_filled.numpy()
                    np.save(os.path.join(self.dataset_path, f'gt_{split}', f'gt_{sample_idx}.npy'), gt)
                    np.save(os.path.join(self.dataset_path, f'mask_{self.name_operator}', f'mask_{sample_idx}.npy'), mask)
                    np.save(os.path.join(self.dataset_path, f'kspace_{self.name_operator}', f'kspace_{sample_idx}.npy'), kspace)
                    
                    if sample_idx < self.n_vis:
                        visualize_data_mri(
                            gt[0], 
                            mask[0], 
                            kspace_full[0], 
                            zero_filled[0], 
                            sample_idx, 
                            self.dataset_path, 
                            self.name_operator, 
                            self.data_config.cmap
                        )
                    sample_idx += 1
                    
        self.logger.info('Data generation completed.')

        