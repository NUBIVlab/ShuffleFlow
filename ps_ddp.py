import gc
import logging
import os
import warnings
from time import time

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0 8.6 9.0'

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group, init_process_group

from utils.dataio import *
from utils.logger import setup_logger
from utils.metrics import *
from utils.name import *
from utils.torch import *
from utils.visualization import *

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')


@hydra.main(version_base='1.3', config_path='configs', config_name='posterior_sampling')
def posterior_sampling(config):
    if config.wandb:
        wandb.init(project=config.inverse_problem.operator.name, 
                   group=config.sampler.name, 
                   config=OmegaConf.to_container(config), 
                   reinit="finish_previous")
        config = OmegaConf.create(dict(wandb.config))
    
    # Create result path.
    name_data = get_name_data(config.inverse_problem.dataset)
    name_operator = get_name_operator(config.inverse_problem.operator)
    name_sampler = get_name_sampler(config.sampler)
    result_path = os.path.join(config.result_path, config.inverse_problem.name, name_data, name_operator, config.sampler.name, name_sampler)
    os.makedirs(result_path, exist_ok=True)
    
    # Initialize distributed training.
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"])) # Set the device for this process.
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device_count, device_name = torch.cuda.device_count(), torch.cuda.get_device_name()
    
    # Set up logger.
    logger = setup_logger(config.sampler.name, level=logging.INFO if global_rank == 0 else logging.ERROR)
    logger.info('Start posterior sampling with %s for %s.', config.sampler.name, config.inverse_problem.name)
    logger.info('GPU: %s x %s.', device_count, device_name)
    
    testset = instantiate(config.inverse_problem.dataset, name_data=name_data, name_operator=name_operator, _recursive_=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    # Load pretrained model.
    if config.sampler.name in ['rml-tv', 'ald-tv']:
        model = None
    else:
        name_diffusion = get_name_diffusion_new(config.model)
        model_save_path = os.path.join(config.model_save_path, name_data, name_diffusion)
        checkpoint = torch.load(os.path.join(model_save_path, f'checkpoint_{config.pretrained_epochs}epochs.pth'), weights_only=True)
        model = instantiate(config.model.model_config)
        model.load_state_dict(checkpoint['ema_state_dict'])
        logger.info('Successfully loaded checkpoint from %s epochs.', config.pretrained_epochs)
        model.to(local_rank).eval()
    
    forward_operator = instantiate(config.inverse_problem.operator)
    img_shape = [config.inverse_problem.dataset.num_channels, config.inverse_problem.dataset.img_size, config.inverse_problem.dataset.img_size]
    sampler = instantiate(config.sampler, img_shape=img_shape, net=model, forward_op=forward_operator, device=torch.device(local_rank))
    evaluator = instantiate(config.inverse_problem.evaluator, forward_operator=forward_operator)
    
    for idx, data in enumerate(test_loader):
        for key, value in data.items():
            data[key] = value.to(local_rank) if isinstance(value, torch.Tensor) else value
                
        data_idx = testset.idx_map(idx)
        logger.info(f'Running posterior sampling on test sample {data_idx} ...')
        os.makedirs(os.path.join(result_path, f'{data_idx}'), exist_ok=True)

        torch.cuda.reset_peak_memory_stats()
        
        forward_operator.load_parameters(data)

        t_start = time()
        samples = sampler(observation=data['measurement'], num_samples=config.num_samples//world_size, verbose=config.verbose and global_rank == 0)
        samples = gather_tensors(samples)
        samples = forward_operator.unnormalize(samples)
        t_end = time()
        
        # Aggregate max VRAM usage across all ranks.
        local_max_vram = torch.cuda.max_memory_allocated(device=local_rank) / 1024 ** 3
        total_max_vram = get_global_average(torch.tensor([local_max_vram], device=local_rank), world_size) * world_size
        
        if global_rank == 0:
            with torch.no_grad():
                metrics = evaluator(samples, data, t_end-t_start, total_max_vram)
            evaluator.print_metrics(metrics, title=f' Metric results on sample {data_idx}: ', logger=logger)
            
            # Save log.
            log = {
                'config': OmegaConf.to_container(config, resolve=True), 
                'gpu': device_name, 'device_count': device_count,
                **metrics
            }
            save_log(os.path.join(result_path, f'{data_idx}'), log)
            if config.wandb:
                wandb.log(metrics, step=idx)
            
            # Save results.
            samples = samples.squeeze(1).detach().cpu().numpy()
            mean, std = samples.mean(0), samples.std(0)
            results = {'mean': mean, 'std': std, 'samples': samples}
            torch.save(results, os.path.join(result_path, f'{data_idx}', f'results_{config.num_samples}samples.pth'))
            
            # Visualization.
            if config.inverse_problem.name == 'single-coil_mri':
                visualize_samples_mri(os.path.join(result_path, f'{data_idx}'), data.copy(), results, log, config.num_samples, config.inverse_problem.dataset.cmap)
            elif 'deblur' in config.inverse_problem.name:
                visualize_samples_deblur(os.path.join(result_path, f'{data_idx}'), data.copy(), results, log, config.num_samples, config.inverse_problem.dataset.cmap)
            elif config.inverse_problem.name == 'phase_retrieval':
                visualize_samples_fpr(os.path.join(result_path, f'{data_idx}'), data.copy(), results, log, config.num_samples, config.inverse_problem.dataset.cmap)
            visualize_calibration(os.path.join(result_path, f'{data_idx}'), data.copy(), results)
        
            logger.info('Results saved to %s.', os.path.join(result_path, f'{data_idx}'))

            # Free GPU memory for next iteration: run GC then release cached allocator memory.
            del mean, std, results, metrics, log, data
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate the metrics.
    if global_rank == 0:
        metric_aggregated = evaluator.aggregate()
        evaluator.print_metrics(metric_aggregated, title=' Aggregated metric results: ', logger=logger)
        if config.wandb:
            wandb.log(metric_aggregated)
            wandb.finish()

    destroy_process_group()

if __name__ == '__main__':
    posterior_sampling()
    