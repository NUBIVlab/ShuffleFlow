import copy
import logging
import math
import os
import warnings
from time import time

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0 8.6 9.0'

import hydra
import torch
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.logger import setup_logger
from utils.name import *
from utils.torch import *
from utils.visualization import plot_loss_curve

warnings.filterwarnings('ignore', category=UserWarning, message='Grad strides do not match bucket view strides')


@hydra.main(version_base='1.3', config_path='configs', config_name='train_diffusion')
def train(config):
    """Training function for diffusion model with Distributed Data Parallel (DPP)."""
    optim_config = config.model.optim_config
    
    # Create model save path.
    name_data = get_name_data(config.inverse_problem.dataset)
    name_diffusion = get_name_diffusion_new(config.model)
    model_save_path = os.path.join(config.model_save_path, name_data, name_diffusion)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize distributed training.
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device_count, device_name = torch.cuda.device_count(), torch.cuda.get_device_name()

    # Set up logger.
    logger = setup_logger('Diffusion training', level=logging.INFO if global_rank == 0 else logging.ERROR)
    logger.info('Start training Diffusion Model on %s dataset (%sx%s) for %s epochs.', config.inverse_problem.dataset.name, config.inverse_problem.dataset.img_size, config.inverse_problem.dataset.img_size, config.n_epochs)
    logger.info('GPU: %s x %s.', device_count, device_name)

    # Create data loaders.
    train_set = instantiate(config.inverse_problem.train_dataset, name_data=name_data, split='train')
    val_set = instantiate(config.inverse_problem.train_dataset, name_data=name_data, split='val')
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=global_rank, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=optim_config.batch_size//world_size, shuffle=False, sampler=train_sampler, num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_set, batch_size=optim_config.batch_size//world_size, shuffle=False, sampler=val_sampler, num_workers=config.num_workers, pin_memory=config.pin_memory)

    # Initialize model.
    model = instantiate(config.model.model_config).to(local_rank)
    if global_rank == 0:
        ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(step, optim_config.warmup_steps) / optim_config.warmup_steps)
    loss_fn = instantiate(config.model.loss_config)
    num_params = get_total_params(model)
    logger.info('Total number of parameters: %d.', num_params)
    
    # Load checkpoint if specified.
    pretrained_epochs = config.pretrained_epochs
    if config.pretrained_epochs > 0:
        try:
            checkpoint = torch.load(os.path.join(model_save_path, f'checkpoint_{pretrained_epochs}epochs.pth'), weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            if global_rank == 0:
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            train_loss_list = checkpoint['train_loss']
            val_loss_list = checkpoint['val_loss']
            val_loss_min = checkpoint['val_loss_min']
            epoch_min = checkpoint['epoch_min']
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            pretrained_steps = checkpoint['steps']
            pretrain_time = checkpoint['time']
            logger.info('Successfully loaded checkpoint from %s epochs.', pretrained_epochs)
        except:
            pretrained_epochs = 0
            pretrained_steps = 0
            logger.warning('Checkpoint not found. Starting training from scratch.')     
    if pretrained_epochs == 0:
        train_loss_list, val_loss_list = [], []
        val_loss_min, epoch_min = math.inf, 0
        pretrain_time = 0.0
        pretrained_steps = 0
        
    # Wrap model with DDP.
    model = DDP(model, device_ids=[local_rank])
            
    # Training Loop.
    steps = pretrained_steps
    t_start = time() - pretrain_time
    for epoch in range(pretrained_epochs+1, pretrained_epochs+config.n_epochs+1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        train_loss = 0.0
        for idx, x in enumerate(train_loader):
            model.train()
            loss = loss_fn(model, x.to(local_rank)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=optim_config.clip)
            optimizer.step()
            scheduler.step()
            
            # Update EMA model.
            if global_rank == 0:
                ema_halflife_nimg = optim_config.ema_halflife_nimg
                curr_nimg = steps * optim_config.batch_size
                ema_halflife_nimg = min(ema_halflife_nimg, curr_nimg * optim_config.ema_rampup_ratio)
                ema_decay = 0.5 ** (optim_config.batch_size / max(ema_halflife_nimg, 1))
                raw_model = unwrap_model(model)
                update_ema(ema_model, raw_model, decay=ema_decay)
            
            train_loss += get_global_average(loss, world_size)
            steps += 1
            
            # Evaluate on validation set.
            if (idx+1) % 25 == 0 or idx+1 == len(train_loader):
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for x in val_loader:
                        loss = loss_fn(model, x.to(local_rank)).mean()
                        val_loss += get_global_average(loss, world_size)
                if idx + 1 == len(train_loader):
                    logger.info(f'[Epoch {epoch}: {idx+1}/{len(train_loader)}]   train_loss={train_loss/len(train_loader):.4e}   val_loss={val_loss/len(val_loader):.4e}   steps={steps}   time={(time()-t_start)/3600:.2f}h')
                else:
                    logger.info(f'[Epoch {epoch}: {idx+1}/{len(train_loader)}]   train_loss={loss.item():.4e}   val_loss={val_loss/len(val_loader):.4e}   steps={steps}   time={(time()-t_start)/3600:.2f}h')
                    
        train_loss_list.append(train_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))
        
        if val_loss_min > val_loss:
            val_loss_min = val_loss
            epoch_min = epoch
        
        # Aggregate max VRAM usage across all ranks.
        local_max_vram = torch.cuda.max_memory_allocated(device=local_rank) / 1024 ** 3
        total_max_vram = get_global_average(torch.tensor([local_max_vram], device=local_rank), world_size) * world_size

        # Plot loss curve and save checkpoint.
        if global_rank == 0:
            plot_loss_curve(model_save_path, train_loss_list, val_loss_list, epoch_min)
            if epoch % config.save_interval == 0:
                checkpoint = {
                    'config': OmegaConf.to_container(config, resolve=True),
                    'num_params': num_params,
                    'world_size': world_size,
                    'epoch': epoch,
                    'steps': steps,
                    'model_state_dict': raw_model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss_list,
                    'val_loss': val_loss_list,
                    'val_loss_min': val_loss_min,
                    'epoch_min': epoch_min,
                    'gpu': device_name,
                    'device_count': device_count,
                    'max_vram': total_max_vram,
                    'time': time() - t_start,
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }
                torch.save(checkpoint, os.path.join(model_save_path, f'checkpoint_{epoch}epochs.pth'))
                logger.info('Checkpoint saved to %s', os.path.join(model_save_path, f'checkpoint_{epoch}epochs.pth'))
        
    destroy_process_group()


if __name__ == '__main__':
    train()