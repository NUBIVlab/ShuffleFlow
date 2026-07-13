import glob
import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from utils.name import *
from utils.torch import *


class fastMRIDataset(Dataset):
    def __init__(self, data_path: str, group: str, name_data: str, split: str='train', **kwargs):
        super().__init__()
        self.dataset_path = os.path.join(data_path, group, name_data)
        self.gt_path = os.path.join(self.dataset_path, 'gt_train')
        files = sorted(os.listdir(self.gt_path))
        num_train = int(0.9 * len(files))
        files = files[:num_train] if split == 'train' else files[num_train:]
        self.num_samples = len(files)
        
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=(0.5, ), std=(0.5, )),
            transforms.RandomHorizontalFlip()
        ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        gt = self.transforms(torch.from_numpy(np.load(os.path.join(self.gt_path, f'gt_{idx}.npy'))))
        print(gt.min(), gt.max())
        return gt


class AFHQv2Dataset(Dataset):
    def __init__(self, data_path: str, group: str, img_size: int, num_channels: int, split: str='train', **kwargs):
        super().__init__()
        if group == 'all':
            cat_files = glob.glob(os.path.join(data_path, 'train', 'cat', '*.png'))
            dog_files = glob.glob(os.path.join(data_path, 'train', 'dog', '*.png'))
            wild_files = glob.glob(os.path.join(data_path, 'train', 'wild', '*.png'))
            self.files = cat_files + dog_files + wild_files
        else:
            self.files = glob.glob(os.path.join(data_path, 'train', group, '*.png'))
        
        num_train = int(0.9 * len(self.files))
        self.files = self.files[:num_train] if split == 'train' else self.files[num_train:]
        self.num_samples = len(self.files)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1) if num_channels==1 else transforms.Lambda(lambda x: x),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=(0.5, ), std=(0.5, )),
            transforms.RandomHorizontalFlip(),
        ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.transforms(cv2.cvtColor(cv2.imread(self.files[idx]), cv2.COLOR_BGR2RGB))


class GaussianDataset(Dataset):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, num_samples: int, **kwargs):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.num_samples = num_samples
        self.mvn = MultivariateNormal(loc=mean.flatten(), covariance_matrix=cov)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.mvn.sample((1,))

# class CelebAHQDataset(TrainingDataset):
#     def __init__(self, config, split: str='train'):
#         super().__init__()
#         if config.data.group == 'all':
#             male_files = glob.glob(os.path.join(config.data.data_path, split, 'male', '*.jpg'))
#             female_files = glob.glob(os.path.join(config.data.data_path, split, 'female', '*.jpg'))
#             self.files = male_files + female_files
#         else:
#             self.files = glob.glob(os.path.join(config.data.data_path, split, config.data.group, '*.jpg'))
#         self.files.sort()
#         self.num_samples = len(self.files)
        
#         self.transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Grayscale(num_output_channels=1) if config.data.num_channels==1 else transforms.Lambda(lambda x: x),
#             transforms.Resize((config.data.img_size, config.data.img_size)),
#             transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#         ])
        
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         gt = self.transforms(cv2.imread(self.files[idx]))
#         return gt


# def get_data_loader_train(config, train: bool=True):
#     if config.data.name == 'fastmri':
#         if train:
#             train_set = get_dataset(name=config.data.name, config=config, split='train')
#             train_set, val_set = random_split(train_set, [int(0.9 * len(train_set)), len(train_set) - int(0.9 * len(train_set))])
#             train_loader = DataLoader(train_set, batch_size=config.model.optim.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             val_loader = DataLoader(val_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return train_loader, val_loader
#         else:
#             test_set = get_dataset(name=config.data.name, config=config.data, split='val')
#             test_loader = DataLoader(test_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return test_loader
#     elif config.data.name == 'afhq-v2':
#         if train:
#             train_set = get_dataset(name=config.data.name, config=config, split='train')
#             train_set, val_set = random_split(train_set, [int(0.9 * len(train_set)), len(train_set) - int(0.9 * len(train_set))])
#             train_loader = DataLoader(train_set, batch_size=config.model.optim.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             val_loader = DataLoader(val_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return train_loader, val_loader
#         else:
#             test_set = get_dataset(name=config.data.name, config=config, split='test')
#             test_loader = DataLoader(test_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return test_loader
#     elif config.data.name == 'celeba-hq':
#         if train:
#             train_set = get_dataset(name=config.data.name, config=config, split='train')
#             train_set, val_set = random_split(train_set, [int(0.9 * len(train_set)), len(train_set) - int(0.9 * len(train_set))])
#             train_loader = DataLoader(train_set, batch_size=config.model.optim.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             val_loader = DataLoader(val_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return train_loader, val_loader
#         else:
#             test_set = get_dataset(name=config.data.name, config=config, split='test')
#             test_loader = DataLoader(test_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return test_loader
#     elif config.data.name == 'celeba':
#         transforms_celeba = transforms.Compose([transforms.ToTensor(), 
#                                                 transforms.Resize((config.data.img_size, config.data.img_size)), 
#                                                 transforms.Grayscale(),
#                                                 transforms.Lambda(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)])
#         if train:
#             train_set = datasets.CelebA(root=config.data.data_path, split='train', download=False, transform=transforms_celeba)
#             val_set = datasets.CelebA(root=config.data.data_path, split='valid', download=False, transform=transforms_celeba)
#             train_loader = DataLoader(train_set, batch_size=config.model.optim.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             val_loader = DataLoader(val_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return train_loader, val_loader
#         else:
#             test_set = datasets.CelebA(root=config.data.data_path, split='test', download=False, transform=transforms_celeba)
#             test_loader = DataLoader(test_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return test_loader
#     elif config.data.name == 'mnist':
#         transforms_mnist = transforms.Compose([transforms.ToTensor(), 
#                                                transforms.Resize((config.data.img_size, config.data.img_size)), 
#                                                transforms.Lambda(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)])
#         if train:
#             train_set = datasets.MNIST(root=config.data.data_path, train=True, download=False, transform=transforms_mnist)
#             train_set, val_set = random_split(train_set, [int(0.9 * len(train_set)), len(train_set) - int(0.9 * len(train_set))])
#             train_loader = DataLoader(train_set, batch_size=config.model.optim.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             val_loader = DataLoader(val_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return train_loader, val_loader
#         else:
#             test_set = datasets.MNIST(root=config.data.data_path, train=False, download=False, transform=transforms_mnist)
#             test_loader = DataLoader(test_set, batch_size=config.model.optim.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
#             return test_loader
#     else:
#         raise ValueError('Invalid dataset.')