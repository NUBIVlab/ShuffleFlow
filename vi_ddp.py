import copy
import gc
import logging
import os
import warnings
from time import time

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0 8.6 9.0'

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from flows import ForwardWrapper, alpha_divergence
from models import get_model
from utils.dataio import *
from utils.logger import setup_logger
from utils.metrics import *
from utils.name import *
from utils.sde import *
from utils.torch import *
from utils.visualization import *

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')
warnings.filterwarnings('ignore', category=UserWarning, message='Grad strides do not match bucket view strides')


def prepare_vi_training(config, img_shape, local_rank, forward_operator, prior):
    optim_config = config.flow.optim_config
    flow = instantiate(config.flow, img_shape=img_shape).to(local_rank)
    ema_flow = copy.deepcopy(flow).eval().requires_grad_(False).to(local_rank)
    forward_wrapper = ForwardWrapper(
        loss_config=config.flow.loss_config,
        prior_config=config.prior,
        data_config=config.inverse_problem.dataset,
        flow=flow,
        forward_operator=forward_operator,
        prior=prior,
    )
    forward_wrapper = DDP(
        forward_wrapper,
        device_ids=[local_rank],
        find_unused_parameters=config.prior.name == 'sds',
    )
    optimizer = Adam(flow.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    return flow, ema_flow, forward_wrapper, optimizer


@hydra.main(version_base='1.3', config_path='configs', config_name='variational_inference')
def variational_inference(config):
    if config.wandb:
        wandb.init(project=config.inverse_problem.operator.name, 
                    group=config.sampler.name, 
                    config=OmegaConf.to_container(config), 
                    reinit="finish_previous")
        config = OmegaConf.create(dict(wandb.config))
    
    optim_config = config.flow.optim_config
    
    # Create result path.
    name_data = get_name_data(config.inverse_problem.dataset)
    name_operator = get_name_operator(config.inverse_problem.operator)
    name_vi = get_name_vi(config)
    result_path = os.path.join(config.result_path, config.inverse_problem.name, name_data, name_operator, config.flow.name, name_vi)
    os.makedirs(result_path, exist_ok=True)
    
    # Initialize distributed training.
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"])) # Set the device for this process.
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device_count, device_name = torch.cuda.device_count(), torch.cuda.get_device_name()

    # Set up logger.
    logger = setup_logger(config.flow.name, level=logging.INFO if global_rank == 0 else logging.ERROR)
    logger.info('Start variational inference with %s for %s.', config.flow.name, config.inverse_problem.name)
    logger.info('GPU: %s x %s.', device_count, device_name)
    if global_rank == 0 and config.auto_resume and not config.save_checkpoint:
        logger.warning('auto_resume is on but save_checkpoint is false: crash recovery will reinitialize from scratch unless you enable checkpoints.')
    
    # Create data loader.
    testset = instantiate(config.inverse_problem.dataset, name_data=name_data, name_operator=name_operator, _recursive_=False)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    
    # Create prior.
    model = None
    if config.prior.name in ['surrogate-score', 'sds', 'l2-score']:
        # Load pretrained model.
        # if config.model.name in ['vp_ncsnpp_256', 'vp_ncsnpp_128']:
        name_diffusion = get_name_diffusion_new(config.model)
        model_save_path = os.path.join(config.model_save_path, name_data, name_diffusion)
        checkpoint = torch.load(os.path.join(model_save_path, f'checkpoint_{config.pretrained_epochs}epochs.pth'), weights_only=True)
        model = instantiate(config.model.model_config)
        model.load_state_dict(checkpoint['ema_state_dict'])
        model = model.model
        # else:
        #     name_diffusion = get_name_diffusion(config)
        #     checkpoint = torch.load(os.path.join(config.model_save_path, name_data, name_diffusion, f'checkpoint_{config.pretrained_epochs}epochs.pth'), weights_only=True)
        #     model = nn.DataParallel(get_model(name=config.model.name, config=config))
        #     model.load_state_dict(checkpoint['model_state_dict'])

        logger.info('Successfully loaded checkpoint from %s epochs.', config.pretrained_epochs)
        model.cuda().eval()
        
    prior = instantiate(config.prior, model=model).to(local_rank)

    img_shape = [config.inverse_problem.dataset.num_channels, config.inverse_problem.dataset.img_size, config.inverse_problem.dataset.img_size]
    forward_operator = instantiate(config.inverse_problem.operator)
    flow_temp = instantiate(config.flow, img_shape=img_shape).to(local_rank)
    # ema_flow = copy.deepcopy(flow).eval().requires_grad_(False).to(local_rank)
    num_params = get_total_params(flow_temp)
    logger.info('Total number of parameters: %d.', num_params)

    evaluator = instantiate(config.inverse_problem.evaluator, forward_operator=forward_operator)
     
    for idx, data in enumerate(test_loader):
        for key, value in data.items():
            data[key] = value.to(local_rank) if isinstance(value, torch.Tensor) else value
                
        data_idx = testset.idx_map(idx)
        logger.info(f'Running variational inference on test sample {data_idx} ...')

        torch.cuda.reset_peak_memory_stats()
        
        forward_operator.load_parameters(data)

        flow, ema_flow, forward_wrapper, optimizer = prepare_vi_training(config, img_shape, local_rank, forward_operator, prior)

        epoch_list, loss_list, likelihood_list, prior_list, entropy_list = [], [], [], [], []
        time_pretrain = 0.0
        last_saved_epoch = 0
        num_failures = 0
        samples = results = metrics = log = mean = std = None
        
        epoch = 0
        t_start = time() - time_pretrain
        while epoch < config.n_epochs:
            epoch += 1
            flow.train()
            z = flow.init_sample(optim_config.batch_size//world_size).to(local_rank)
            likelihood, prior_term, entropy = forward_wrapper(z, data, epoch)
            loss = alpha_divergence(likelihood, prior_term, entropy, config.flow.loss_config.alpha)

            # Check for NaN or ±Inf before backward/step so we never apply NaN gradients.
            val_loss = loss.clone().detach()
            local_any_bad = (torch.isnan(val_loss) | torch.isinf(val_loss)).any().to(dtype=torch.float32).view(1)
            if dist.is_initialized():
                any_bad = bool((gather_tensors(local_any_bad) > 0).any().item())
            else:
                any_bad = bool(local_any_bad.item() > 0)

            ########## Handle training failures. ##########
            if any_bad:
                optimizer.zero_grad(set_to_none=True)
                num_failures += 1
                if dist.is_initialized():
                    dist.barrier()
                
                # Skip current test sample if auto_resume is disabled.
                if not config.auto_resume:
                    if global_rank == 0:
                        logger.error('Non-finite loss at epoch %d (loss=%s); auto_resume is disabled; skipping sample %s.', epoch, val_loss.item(), data_idx)
                    break
                
                # Skip current test sample if maximum resume attempts are exceeded.
                if num_failures > config.max_resume_attempts:
                    if global_rank == 0:
                        logger.error('Maximum resume attempts exceeded; skipping sample %s.', data_idx)
                    break
                
                # Auto-resume training.
                if global_rank == 0:
                    logger.error('Non-finite loss at epoch %d; recovery attempt %d/%d.', epoch, num_failures, config.max_resume_attempts)

                if config.save_checkpoint and last_saved_epoch > 0:
                    try:
                        checkpoint = torch.load(os.path.join(result_path, str(data_idx), f'{last_saved_epoch}epochs', 'checkpoint.pth'), weights_only=True)
                        flow.load_state_dict(checkpoint['model_state_dict'])
                        ema_flow.load_state_dict(checkpoint['ema_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        epoch_list = checkpoint['epoch_list']
                        loss_list = checkpoint['loss_list']
                        likelihood_list = checkpoint['likelihood_list']
                        prior_list = checkpoint['prior_list']
                        entropy_list = checkpoint['entropy_list']
                        time_pretrain = float(checkpoint['time'])
                        t_start = time() - time_pretrain
                        epoch = last_saved_epoch
                        if global_rank == 0:
                            logger.info('Successfully restored to epoch %d; continuing.', last_saved_epoch)
                        continue
                    except Exception as e:
                        if global_rank == 0:
                            logger.warning('Failed to load checkpoint (%s); reinitializing from scratch.', e)

                # Initialize training.
                if global_rank == 0:
                    logger.warning('No available checkpoint for sample %s; reinitializing flow and optimizer from scratch.', data_idx)
                del forward_wrapper
                gc.collect()
                torch.cuda.empty_cache()
                flow, ema_flow, forward_wrapper, optimizer = prepare_vi_training(config, img_shape, local_rank, forward_operator, prior)
                epoch_list, loss_list, likelihood_list, prior_list, entropy_list = [], [], [], [], []
                time_pretrain = 0.0
                t_start = time()
                epoch = 0
                last_saved_epoch = 0
                continue
            ########## End of training failures. ##########

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), optim_config.clip)
            optimizer.step()

            # Update EMA flow.
            if optim_config.ema:
                decay = min(optim_config.ema_decay, (epoch + 1) / (epoch + 10))
                raw_flow = unwrap_model(flow)
                update_ema(ema_flow, raw_flow, decay)

            epoch_list.append(epoch)
            loss_list.append(get_global_average(loss, world_size))
            likelihood_list.append(get_global_average(likelihood.mean(), world_size))
            prior_list.append(get_global_average(prior_term.mean(), world_size))
            entropy_list.append(get_global_average(entropy.mean(), world_size))

            if epoch % 25 == 0:
                logger.info(f"[{epoch}/{config.n_epochs}]   loss={loss_list[-1]:.3e}  likelihood={likelihood_list[-1]:.3e}  prior={prior_list[-1]:.3e}  entropy={entropy_list[-1]:.3e}  time={(time()-t_start)/3600:.2f}h")

            del loss, likelihood, prior_term, entropy

            ########## Evaluation and saving results. ##########
            if epoch % config.eval_interval == 0:
                t_end = time()
                
                # Draw samples to calculate mean and std.
                eval_flow = ema_flow if optim_config.ema else flow
                forward_wrapper.eval()
                with torch.no_grad():
                    z = eval_flow.init_sample(config.num_samples//world_size).to(local_rank)
                    samples, _ = eval_flow(z)
                    samples = samples.view(-1, *img_shape)
                    samples = gather_tensors(samples)
                    samples = forward_operator.unnormalize(samples)
                    
                # Aggregate max VRAM usage across all ranks.
                local_max_vram = torch.cuda.max_memory_allocated(device=local_rank) / 1024 ** 3
                total_max_vram = get_global_average(torch.tensor([local_max_vram], device=local_rank), world_size) * world_size

                if global_rank == 0:
                    os.makedirs(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), exist_ok=True)
                    
                    with torch.no_grad():
                        metrics = evaluator(samples, data, t_end-t_start, total_max_vram)
                    evaluator.print_metrics(metrics, title=f' Metric results on sample {data_idx}: ', logger=logger)
                    
                    # Save log.
                    log = {
                        'config': OmegaConf.to_container(config, resolve=True), 
                        'num_params': num_params,
                        'world_size': world_size,
                        'gpu': device_name, 
                        'device_count': device_count,
                        **metrics
                    }
                    save_log(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), log)
                    if config.wandb:
                        wandb.log(metrics, step=idx)
                    
                    # Save results.
                    samples = samples.squeeze(1).detach().cpu().numpy()
                    mean, std = samples.mean(0), samples.std(0)
                    results = {'mean': mean, 'std': std, 'samples': samples}
                    torch.save(results, os.path.join(result_path, str(data_idx), f'{epoch}epochs', f'results_{config.num_samples}samples.pth'))
                    
                    # Visualization.
                    visualize_loss_curve_vi(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), epoch_list, loss_list, likelihood_list, prior_list, entropy_list)
                    if config.inverse_problem.name == 'single-coil_mri':
                        visualize_samples_mri(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), data.copy(), results, log, config.num_samples, config.inverse_problem.dataset.cmap)
                    elif 'deblur' in config.inverse_problem.name:
                        visualize_samples_deblur(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), data.copy(), results, log, config.num_samples, config.inverse_problem.dataset.cmap)
                    elif config.inverse_problem.name == 'phase_retrieval':
                        visualize_samples_fpr(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), data.copy(), results, log, config.num_samples, config.inverse_problem.dataset.cmap)
                    visualize_calibration(os.path.join(result_path, str(data_idx), f'{epoch}epochs'), data.copy(), results)
                    
                    # Save checkpoint.
                    if config.save_checkpoint:
                        checkpoint = {
                            'config': OmegaConf.to_container(config, resolve=True),
                            'epoch': epoch,
                            'num_params': num_params,
                            'world_size': world_size,
                            'model_state_dict': unwrap_model(flow).state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'ema_state_dict': ema_flow.state_dict(),
                            'epoch_list': epoch_list, 
                            'loss_list': loss_list, 
                            'likelihood_list': likelihood_list, 
                            'prior_list': prior_list, 
                            'entropy_list': entropy_list,
                            'time': t_end - t_start,
                            'max_vram': total_max_vram,
                            'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        }
                        torch.save(checkpoint, os.path.join(result_path, str(data_idx), f'{epoch}epochs', 'checkpoint.pth'))
                        
                    logger.info('Results saved to %s.', os.path.join(result_path, str(data_idx), f'{epoch}epochs'))

                if config.save_checkpoint and dist.is_initialized():
                    dist.barrier()
                if config.save_checkpoint:
                    last_saved_epoch = epoch
            ########## End of evaluation and saving results. ##########

        # Free GPU memory for next iteration: run GC then release cached allocator memory.
        del data
        if samples is not None:
            del samples
        if global_rank == 0 and results is not None:
            del results, metrics, log, mean, std
        gc.collect()
        torch.cuda.empty_cache()
                
    del forward_wrapper, flow, prior, optimizer
    if config.prior.name in ['logpbound', 'sds', 'l2-score']:
        del model
    torch.cuda.empty_cache()
        
    # Aggregate the metrics.
    metric_aggregated = evaluator.aggregate()
    evaluator.print_metrics(metric_aggregated, title=' Aggregated metric results: ', logger=logger)
    if config.wandb:
        wandb.log(metric_aggregated)
        wandb.finish()

    destroy_process_group()


if __name__ == '__main__':
    variational_inference()
