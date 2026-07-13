import glob
import json
import os

import cv2
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from forward_operators import Deblur
from utils.visualization import visualize_data_deblur

from . import BaseDataGenerator


class DeblurDataGenerator(BaseDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data_path = os.path.join(self.data_config.data_path, 'test', self.data_config.group)
        self.dataset_path = os.path.join(self.data_config.data_path, 'test', self.name_data)
        if self.data_config.group == 'all':
            cat_files = glob.glob(os.path.join(self.data_config.data_path, 'test', 'cat', '*.png'))
            dog_files = glob.glob(os.path.join(self.data_config.data_path, 'test', 'dog', '*.png'))
            wild_files = glob.glob(os.path.join(self.data_config.data_path, 'test', 'wild', '*.png'))
            self.files = cat_files + dog_files + wild_files
        else:
            self.files = glob.glob(os.path.join(self.data_config.data_path, 'test', self.data_config.group, '*.png'))
        self.files.sort()
        
        # Create directories.
        os.makedirs(os.path.join(self.dataset_path, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, f'psf_{self.name_operator}'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, f'obs_{self.name_operator}'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, f'vis_{self.name_operator}'), exist_ok=True)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1) if self.data_config.num_channels==1 else transforms.Lambda(lambda x: x),
            transforms.Resize((self.data_config.img_size, self.data_config.img_size)),
        ])
        
        # # # Save dataset information.
        # info_file = os.path.join(self.dataset_path, f'vis_{self.name_operator}', 'info.json')
        # # info = {'config': OmegaConf.to_container(self.inverse_problem_config, resolve=True), }
        # # with open(info_file, 'w') as f:
        # #     json.dump(info, f)
        # self.logger.info('Dataset information saved to %s.', info_file)

    def generate_data(self):
        self.logger.info('Start generating data for %s using %s dataset.', self.operator_config.name, self.data_config.name)

        for sample_idx, file in tqdm(enumerate(self.files), desc=f'Generating data', total=len(self.files)):
            gt = cv2.imread(file)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = self.transforms(gt)
            if self.data_config.normalize:
                gt = (gt - gt.min()) / (gt.max() - gt.min())
            if self.operator_config.name == 'gaussian_deblur':
                psf = Deblur.get_gaussian_kernel(gt.shape[-2:], self.operator_config.sigma_kernel)
            elif self.operator_config.name == 'motion_deblur':
                psf = Deblur.get_motion_kernel(gt.shape[-2:], self.operator_config.intensity)
            else:
                raise NotImplementedError(f'Unsupported deblur type: {self.operator_config.name}.')
                
            obs = Deblur(psf=psf, sigma_noise=self.operator_config.sigma_noise)(gt)
            gt, psf, obs = gt.numpy(), psf.numpy(), obs.numpy()
            np.save(os.path.join(self.dataset_path, 'gt', f'gt_{sample_idx}.npy'), gt)
            np.save(os.path.join(self.dataset_path, f'psf_{self.name_operator}', f'psf_{sample_idx}.npy'), psf)
            np.save(os.path.join(self.dataset_path, f'obs_{self.name_operator}', f'obs_{sample_idx}.npy'), obs)
            
            if sample_idx < self.n_vis:
                visualize_data_deblur(
                    gt.transpose(1,2,0) if gt.shape[-1] == 3 else gt[0], 
                    psf[0], 
                    obs.transpose(1,2,0) if obs.shape[-1] == 3 else obs[0], 
                    sample_idx, 
                    self.dataset_path, 
                    self.name_operator, 
                    self.data_config.cmap
                )
