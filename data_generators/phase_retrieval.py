import glob
import json
import os

import cv2
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from forward_operators import PhaseRetrieval
from utils.phase_retrieval import *
from utils.visualization import visualize_data_pr

from . import BaseDataGenerator


class PhaseRetrievalDataGenerator(BaseDataGenerator):
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
        os.makedirs(os.path.join(self.dataset_path, f'amp_{self.name_operator}'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, f'vis_{self.name_operator}'), exist_ok=True)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1) if self.data_config.num_channels==1 else transforms.Lambda(lambda x: x),
            transforms.Resize((self.data_config.img_size, self.data_config.img_size)),
        ])
        

    def generate_data(self):
        self.logger.info('Start generating data for %s using %s dataset.', self.operator_config.name, self.data_config.name)

        for sample_idx, file in tqdm(enumerate(self.files), desc=f'Generating data', total=len(self.files)):
            gt = cv2.imread(file)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = self.transforms(gt)
            if self.data_config.normalize:
                gt = (gt - gt.min()) / (gt.max() - gt.min())

            amp = PhaseRetrieval(oversample=self.operator_config.oversample).image_to_amplitude(gt, oversample=self.operator_config.oversample)
            ifft = PhaseRetrieval(oversample=self.operator_config.oversample).amplitude_to_image(amp, crop=get_pad(self.operator_config.oversample, gt.shape[-1]))
            print(amp.max(), amp.min(), amp.mean())
            assert False
            gt, amp, ifft = gt.numpy(), amp.numpy(), ifft.numpy()
            np.save(os.path.join(self.dataset_path, 'gt', f'gt_{sample_idx}.npy'), gt)
            np.save(os.path.join(self.dataset_path, f'amp_{self.name_operator}', f'amp_{sample_idx}.npy'), amp)

            if sample_idx < self.n_vis:
                visualize_data_pr(
                    gt.transpose(1,2,0) if gt.shape[-1] == 3 else gt[0], 
                    amp.transpose(1,2,0) if amp.shape[-1] == 3 else amp[0], 
                    ifft.transpose(1,2,0) if ifft.shape[-1] == 3 else ifft[0], 
                    sample_idx, 
                    self.dataset_path, 
                    self.name_operator, 
                    self.data_config.cmap
                )
