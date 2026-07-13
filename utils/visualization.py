import os

import numpy as np
from hydra.utils import instantiate
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.config import *
from utils.dataio import *
from utils.metrics import *
from utils.name import *

FONT_WEIGHT = 'semibold'
CMAP_STD = 'PuBu'
CMAP_ERROR = 'PuRd'
CMAP_RATIO = 'Purples'
plt.rcParams.update({'font.weight': FONT_WEIGHT, 'font.family': 'DejaVu Serif'})


def plot_loss_curve(model_save_path, train_loss, val_loss, epoch_min):
    n_epochs = len(train_loss)
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, n_epochs+1), train_loss, '-o', markersize=4, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_loss, '-o', markersize=4, label='Valid Loss')
    plt.plot([epoch_min], [val_loss[epoch_min-1]], 'ro', markersize=7, label='Best Epoch')
    plt.title('Loss Curve', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=15)
    file_name = os.path.join(model_save_path, 'loss_curve.jpg')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def visualize_loss_curve_vi(result_path, epoch_list, loss_list, likelihood_list, prior_list, entropy_list):
    # Loss curve.
    fig = plt.figure(figsize=(9, 5))
    plt.plot(epoch_list, loss_list, label='Loss', color='black')
    plt.plot(epoch_list, likelihood_list, label='Data likelihood', color='red')
    plt.plot(epoch_list, prior_list, label='Prior', color='blue')   
    plt.plot(epoch_list, entropy_list, label='Entropy', color='green')
    plt.yscale('symlog', linthresh=1e3)
    plt.xlim(0, epoch_list[-1])
    plt.title('Loss Curve', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(result_path, 'loss_curve.png'), bbox_inches='tight')
    plt.close()


def visualize_calibration(result_path, data, results):
    for key, value in data.items():
        data[key] = value.squeeze().cpu().detach().numpy() if isinstance(value, torch.Tensor) else value
    
    abs_error = np.abs(results['mean'] - data['gt']).flatten()
    std = results['std'].flatten()
    
    fig = plt.figure(figsize=(8, 8))
    stds = np.linspace(0, std.max(), 100)
    plt.plot(stds, stds, linestyle='--', color='r', label='y=x', linewidth=1.5)
    plt.plot(stds, 3*stds, linestyle='--', color='r', label='y=3x', linewidth=1.5)
    plt.scatter(std, abs_error, s=5)
    plt.xlabel('Standard deviation ($\sigma$)', fontsize=14)
    plt.ylabel('Absolute error ($|\mu-t|$)', fontsize=14)
    plt.xlim(0, std.max())
    plt.ylim(0, abs_error.max())
    
    plt.title('Uncertainty calibration', fontsize=18)
    plt.savefig(os.path.join(result_path, 'calibration.png'), bbox_inches='tight')
    plt.close()
     
        
def visualize_data_mri(gt, mask, kspace_full, zero_filled, idx, dataset_path, name_operator, cmap, vmax=None):
    kspace_full = np.abs(kspace_full[...,0] + 1j*kspace_full[...,1])

    fig = plt.figure(figsize=(24.5, 5.5))
    gs = plt.GridSpec(4, 20)
    norm_img = Normalize(vmin=0.0, vmax=np.quantile(gt, 0.99) if vmax is None else vmax)
    norm_mask = Normalize(vmin=0, vmax=1.0)
    norm_kspace = LogNorm(vmin=1e-4, vmax=kspace_full.max())
    
    # Ground Truth.
    ax = plt.subplot(gs[0:4, 0:4])
    ax.imshow(gt, cmap=cmap, norm=norm_img)
    ax.set_title('Ground Truth', fontsize=16, fontweight=FONT_WEIGHT)
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')
    
    # K-space.
    ax = plt.subplot(gs[0:4, 5:9])
    ax.imshow(kspace_full, cmap='gray', norm=norm_kspace)
    ax.set_title(r'$\kappa$-space', fontsize=16, fontweight=FONT_WEIGHT)
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_kspace)
    
    # Mask.
    ax = plt.subplot(gs[0:4, 10:14])
    ax.imshow(mask, cmap='gray', norm=norm_mask)
    ax.set_title('Mask', fontsize=16, fontweight=FONT_WEIGHT)
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_mask)
    
    # Zero-filled IFFT.
    ax = plt.subplot(gs[0:4, 15:19])
    ax.imshow(zero_filled, cmap=cmap, norm=norm_img)
    ax.set_title('Zero-filled IFFT', fontsize=16, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={psnr(gt, zero_filled):.2f}', x=int(gt.shape[-1]*0.02), y=int(gt.shape[-2]*0.07), fontsize=14, color='white')
    ax.text(s=f'SSIM={ssim(gt, zero_filled):.3f}', x=int(gt.shape[-1]*0.53), y=int(gt.shape[-2]*0.07), fontsize=14, color='white')
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')

    plt.savefig(os.path.join(dataset_path, f'vis_{name_operator}', f'vis_{idx}.png'), bbox_inches='tight', dpi=128)
    plt.close()


                
def visualize_data_deblur(gt, psf, obs, idx, dataset_path, name_operator, cmap='gray'):
    vmin = 1e-7
    psf = psf.clip(min=vmin)
    obs = obs.clip(min=0.0, max=1.0)
    
    fig = plt.figure(figsize=(24, 5.5))
    gs = plt.GridSpec(4, 20)
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    norm_psf = LogNorm(vmin=vmin, vmax=psf.max())
    
    # Ground Truth.
    ax = plt.subplot(gs[0:4, 0:4])
    if gt.shape[-1] == 3:
        ax.imshow(gt)
    else:
        ax.imshow(gt, cmap=cmap, norm=norm_img)
        cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
        cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img)
    ax.set_title('Ground Truth', fontsize=16, fontweight=FONT_WEIGHT)
    ax.axis('off')

    # PSF.
    ax = plt.subplot(gs[0:4, 5:9])
    ax.imshow(psf, cmap=cmap)#, norm=norm_psf)
    ax.set_title('PSF', fontsize=16, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax)#, norm=norm_psf)
    
    # Observation.
    ax = plt.subplot(gs[0:4, 10:14])
    if obs.shape[-1] == 3:
        ax.imshow(obs)
    else:
        ax.imshow(obs, cmap=cmap, norm=norm_img)
        cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
        cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img)
    ax.text(s=f'PSNR={psnr(gt, obs):.2f}', x=int(gt.shape[-1]*0.02), y=int(gt.shape[-2]*0.07), fontsize=14, color='white')
    ax.text(s=f'SSIM={ssim(gt, obs):.3f}', x=int(gt.shape[-1]*0.53), y=int(gt.shape[-2]*0.07), fontsize=14, color='white')
    ax.set_title('Obs', fontsize=16, fontweight=FONT_WEIGHT)
    ax.axis('off')

    plt.savefig(os.path.join(dataset_path, f'vis_{name_operator}', f'vis_{idx}.png'), bbox_inches='tight', dpi=128)
    plt.close()


def visualize_data_pr(gt, amp, ifft, idx, dataset_path, name_operator, cmap, vmax=None):
    
    fig = plt.figure(figsize=(24, 5.5))
    gs = plt.GridSpec(4, 20)
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    norm_amp = LogNorm(vmin=1e-4, vmax=amp.max())
    
    # Ground Truth.
    ax = plt.subplot(gs[0:4, 0:4])
    ax.imshow(gt, cmap=cmap, norm=norm_img)
    ax.set_title('Ground Truth', fontsize=16, fontweight=FONT_WEIGHT)
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')
    
    # K-space.
    ax = plt.subplot(gs[0:4, 5:9])
    ax.imshow(amp, cmap='gray', norm=norm_amp)
    ax.set_title('Amplitude', fontsize=16, fontweight=FONT_WEIGHT)
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_amp)
    
    # IFFT.
    ax = plt.subplot(gs[0:4, 10:14])
    ax.imshow(ifft, cmap=cmap, norm=norm_img)
    ax.set_title('IFFT', fontsize=16, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={psnr(gt, ifft):.2f}', x=int(gt.shape[-1]*0.02), y=int(gt.shape[-2]*0.07), fontsize=14, color='white')
    ax.text(s=f'SSIM={ssim(gt, ifft):.3f}', x=int(gt.shape[-1]*0.53), y=int(gt.shape[-2]*0.07), fontsize=14, color='white')
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')

    plt.savefig(os.path.join(dataset_path, f'vis_{name_operator}', f'vis_{idx}.png'), bbox_inches='tight', dpi=128)
    plt.close()
    
    
def visualize_data_cdp(gt, measurements, idx, dataset_path, name_operator, cmap, vmax=None):
    
    fig = plt.figure(figsize=(24, 5.5))
    gs = plt.GridSpec(4, 20)
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    # norm_amp = LogNorm(vmin=1e-4, vmax=amp.max())
    
    # Ground Truth.
    ax = plt.subplot(gs[0:4, 0:4])
    ax.imshow(gt, cmap=cmap, norm=norm_img)
    ax.set_title('Ground Truth', fontsize=16, fontweight=FONT_WEIGHT)
    cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')
    
    # # K-space.
    # ax = plt.subplot(gs[0:4, 5:9])
    # ax.imshow(amp, cmap='gray', norm=norm_amp)
    # ax.set_title('Amplitude', fontsize=16, fontweight=FONT_WEIGHT)
    # cax = fig.add_axes([ax.get_position().x1+5e-3, ax.get_position().y0, 0.01, ax.get_position().height])
    # cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_amp)
    

    plt.savefig(os.path.join(dataset_path, f'vis_{name_operator}', f'vis_{idx}.png'), bbox_inches='tight', dpi=128)
    plt.close()
    

def ablation_figure(config: DictConfig, value_name: str, value_list: list, cmap: str, gs_scale: int, fig_scale: float):

    name_data = get_name_data(config.inverse_problem.dataset)
    name_operator = get_name_operator(config.inverse_problem.operator)
    testset = instantiate(config.inverse_problem.dataset, name_data=name_data, name_operator=name_operator, _recursive_=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data = next(iter(test_loader))
    gt = data['gt'].squeeze().numpy()
    data_idx = testset.idx_map(0)
    
    fig = plt.figure(figsize=(len(value_list)*fig_scale, 5*fig_scale))
    gs = plt.GridSpec(5*gs_scale, len(value_list)*gs_scale, wspace=0.2, hspace=0.2)
    
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    norm_std = Normalize(vmin=0.0, vmax=0.06)
    norm_error = Normalize(vmin=0.0, vmax=0.2)
    norm_ratio = Normalize(vmin=0.0, vmax=4.0)
    
    img_size = gt.shape[-1]

    for idx, value in enumerate(value_list):
        config = set_config_absolute(config, value_name, value)
        if config.name == 'variational_inference':
            results, log = load_results_vi(config, data_idx)
        elif config.name == 'posterior_sampling':
            results, log = load_results_ps(config, data_idx)
        else:
            raise ValueError(f'Unsupported task: {config.name}.')
        
        error = np.abs(gt - results['mean'])
        ratio = error / results['std']
        
        # Mean.
        ax = plt.subplot(gs[0:gs_scale, idx*gs_scale:(idx+1)*gs_scale])
        plt.imshow(results['mean'], cmap=cmap, norm=norm_img)
        plt.xticks([])
        plt.yticks([])
        ax.set_title(f'{value_name[-1]}={value}', fontsize=19, fontweight='bold')
        plt.text(s=f'PSNR={log["psnr"]:.1f}', x=int(img_size*0.02), y=int(img_size/12), fontsize=14, color='white', fontweight='bold')
        plt.text(s=f'SSIM={log["ssim"]:.2f}', x=int(img_size*0.53), y=int(img_size/12), fontsize=14, color='white', fontweight='bold')
        plt.text(s=f'LPIPS={log["lpips"]:.3f}', x=int(img_size*0.02), y=int(img_size*0.97), fontsize=14, color='white', fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r'Mean ($\mu$)', fontsize=19, fontweight='bold')
        if (idx+1) % len(value_list) == 0:
            cax = fig.add_axes([ax.get_position().x1+0.004, ax.get_position().y0, 0.008, ax.get_position().height])
            cb = plt.colorbar(cax=cax, norm=norm_img)
            cb.set_ticks([0,1])
            cb.set_ticklabels(['0.0','1.0'], fontsize=14, fontweight='bold')
            
            
        # Std. dev.
        ax = plt.subplot(gs[gs_scale:2*gs_scale, idx*gs_scale:(idx+1)*gs_scale])
        plt.imshow(results['std'], cmap=CMAP_STD, norm=norm_std)
        plt.xticks([])
        plt.yticks([])
        plt.text(s=f'NLL={log["nll"]:.2f}', x=int(img_size*0.02), y=int(img_size/12), fontsize=14, color='black', fontweight='bold')
        plt.text(s=f'ECE={log["ece"]:.2f}', x=int(img_size*0.53), y=int(img_size/12), fontsize=14, color='black', fontweight='bold')
        if idx == 0:
            ax.set_ylabel(f'Std. dev. ($\sigma$)', fontsize=19, fontweight='bold')
        if (idx+1) % len(value_list) == 0:
            cax = fig.add_axes([ax.get_position().x1+0.004, ax.get_position().y0, 0.008, ax.get_position().height])
            cb = plt.colorbar(cax=cax, norm=norm_std)
            
        # Abs. error.
        ax = plt.subplot(gs[2*gs_scale:3*gs_scale, idx*gs_scale:(idx+1)*gs_scale])
        plt.imshow(error, cmap=CMAP_ERROR, norm=norm_error)
        plt.xticks([])
        plt.yticks([])
        if idx == 0:
            ax.set_ylabel(r'Abs. error ($\vert\mu - t\vert$)', fontsize=19, fontweight='bold')
        if (idx+1) % len(value_list) == 0:
            cax = fig.add_axes([ax.get_position().x1+0.004, ax.get_position().y0, 0.008, ax.get_position().height])
            cb = plt.colorbar(cax=cax, norm=norm_error)
            
        # Ratio.
        ax = plt.subplot(gs[3*gs_scale:4*gs_scale, idx*gs_scale:(idx+1)*gs_scale])
        plt.imshow(ratio, cmap=CMAP_RATIO, norm=norm_ratio)
        ax.scatter(np.where(ratio > 3)[1], np.where(ratio > 3)[0], marker='s', color='red', s=0.55)
        plt.xticks([])
        plt.yticks([])
        plt.text(s=f'OODR={100*log["oodr"]:.1f}%', x=int(img_size*0.02), y=int(img_size/12), fontsize=14, color='black', fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r'Ratio ($\vert\mu - t\vert / \sigma$)', fontsize=19, fontweight='bold')
        if (idx+1) % len(value_list) == 0:  
            cax = fig.add_axes([ax.get_position().x1+0.004, ax.get_position().y0, 0.008, ax.get_position().height])
            cb = plt.colorbar(cax=cax, norm=norm_ratio)
            
        # Calibration curve.
        ax = plt.subplot(gs[4*gs_scale:5*gs_scale, idx*gs_scale:(idx+1)*gs_scale])
        ax.plot(results['std'].ravel(), error.ravel(), 'o', markersize=1.5, color='black')
        ax.plot([0, 1], [0, 1], '--', linewidth=2.5, color='green', label='y=x')
        ax.plot([0, 1], [0, 3], '--', linewidth=2.5, color='red', label='y=3x')
        ax.set_xlim(0, 0.12)
        ax.set_ylim(0, 0.36)
        ax.set_xlabel(f'Std. dev. ($\sigma$)', fontsize=19, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r'Abs. error ($\vert\mu - t\vert$)', fontsize=19, fontweight='bold')
            ax.set_yticks([0.0, 0.1, 0.2, 0.3])
            ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3'], fontsize=15, fontweight='bold')
            ax.legend(loc='upper right', fontsize=16)
        else:
            ax.set_yticks([])
        ax.set_xticks([0.00, 0.05, 0.10])
        ax.set_xticklabels(['0.00', '0.05', '0.10'], fontsize=15, fontweight='bold')


def ablation_samples(config: DictConfig, value_name: str, value_list: list, cmap: str, gs_scale: int, fig_scale: float, num_samples: int=4):
    
    name_data = get_name_data(config.inverse_problem.dataset)
    name_operator = get_name_operator(config.inverse_problem.operator)
    testset = instantiate(config.inverse_problem.dataset, name_data=name_data, name_operator=name_operator, _recursive_=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data = next(iter(test_loader))
    gt = data['gt'].squeeze().numpy()
    data_idx = testset.idx_map(0)
    
    fig = plt.figure(figsize=(num_samples*fig_scale, len(value_list)*fig_scale))
    gs = plt.GridSpec(len(value_list)*gs_scale, num_samples*gs_scale, wspace=0.2, hspace=0.2)
    
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    img_size = gt.shape[-1]

    for idx, value in enumerate(value_list):
        config = set_config_absolute(config, value_name, value)
        if config.name == 'variational_inference':
            results, log = load_results_vi(config, data_idx)
        elif config.name == 'posterior_sampling':
            results, log = load_results_ps(config, data_idx)
        else:
            raise ValueError(f'Unsupported task: {config.name}.')
        
        # Samples.
        samples = results['samples']
        # Randomly shuffle samples along the first dimension
        idx_perm = np.random.permutation(samples.shape[0])
        samples = samples[idx_perm]
 
        for idx_sample in range(num_samples):
            ax = plt.subplot(gs[idx*gs_scale:(idx+1)*gs_scale, idx_sample*gs_scale:(idx_sample+1)*gs_scale])
            plt.imshow(samples[idx_sample], cmap=cmap, norm=norm_img)
            plt.xticks([])
            plt.yticks([])
            if idx_sample == 0:
                ax.set_ylabel(f'{value_name[-1]}={value}', fontsize=20, fontweight='bold')
            plt.text(s=f'PSNR={psnr(gt, samples[idx_sample], data_range=1.0):.1f}', x=int(img_size*0.02), y=int(img_size/12), fontsize=14, color='red', fontweight='bold')
            plt.text(s=f'SSIM={ssim(gt, samples[idx_sample], data_range=1.0):.2f}', x=int(img_size*0.53), y=int(img_size/12), fontsize=14, color='red', fontweight='bold')
            # plt.text(s=f't={log["time"]:.1f}s', x=2, y=123, fontsize=15, color='white', fontweight='bold')
            if idx == 0:
                ax.set_title(f'Sample {idx_sample+1}', fontsize=20, fontweight='bold')
            if idx_sample == num_samples-1:
                cax = fig.add_axes([ax.get_position().x1+0.004, ax.get_position().y0, 0.008, ax.get_position().height])
                cb = plt.colorbar(cax=cax, norm=norm_img)
                cb.set_ticks([0,1])
                cb.set_ticklabels(['0.0','1.0'], fontsize=14, fontweight='bold')
    

def visualize_samples_mri(result_path, data, results, log, num_samples, cmap, vmin=0.0):
    for key, value in data.items():
        data[key] = value.squeeze().cpu().detach().numpy() if isinstance(value, torch.Tensor) else value
        
    fig = plt.figure(figsize=(20, 14))
    norm_img = Normalize(vmin=vmin, vmax=1.0)
    norm_mask = Normalize(vmin=0, vmax=1.0)
    norm_std = Normalize(vmin=0, vmax=results['std'].max())
    norm_error = Normalize(vmin=0, vmax=np.abs(data['gt']-results['mean']).max())
    norm_ratio = Normalize(vmin=0, vmax=4.0)
    
    ax = plt.subplot(3,4,1)
    ax.imshow(data['zero_filled'], cmap=cmap, norm=norm_img)
    ax.set_title('Zero-filled IFFT', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={psnr(data["gt"], data["zero_filled"]):.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'SSIM={ssim(data["gt"], data["zero_filled"]):.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'LPIPS={log["lpips"]:.3f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.97, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')
    
    ax = plt.subplot(3,4,5)
    ax.imshow(data['mask'], cmap='gray', norm=norm_mask)
    ax.set_title('Mask', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_mask)
        
    ax = plt.subplot(3,4,2)
    ax.imshow(results['mean'], cmap=cmap, norm=norm_img)
    ax.set_title(f'Mean $\mu$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={log["psnr"]:.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'SSIM={log["ssim"]:.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'LPIPS={log["lpips"]:.3f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.97, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')

    ax = plt.subplot(3,4,6)
    ax.imshow(data['gt'], cmap=cmap, norm=norm_img)
    ax.set_title('Ground Truth $t$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')
    
    ax = plt.subplot(3,4,3)
    ax.imshow(results['std'], cmap=cmap, norm=norm_std)
    ax.set_title(f'Std. Dev. $\sigma$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'NLL={log["nll"]:.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'ECE={log["ece"]:.2f}', x=data["gt"].shape[-2]*0.6, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_std)
    
    ax = plt.subplot(3,4,7)
    ax.imshow(results['std']/results['mean'], cmap=cmap)
    ax.set_title(f'$\sigma / \mu$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax)
    
    ax = plt.subplot(3,4,4)
    ax.imshow(np.abs(data['gt']-results['mean']), cmap=cmap, norm=norm_error)
    ax.set_title('Absolute error $|\mu - t|$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_error)
    
    ax = plt.subplot(3,4,8)
    ax.imshow(np.abs(data['gt']-results['mean'])/results['std'], cmap=cmap, norm=norm_ratio)
    ax.set_title(f'$|\mu - t| / \sigma$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'OODR={100*log["oodr"]:.2f}%', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_ratio)

    num_samples_show = min(num_samples, 4)
    for idx in range(num_samples_show):
        ax = plt.subplot(3,4,idx+9)
        plt.imshow(results['samples'][idx], cmap=cmap, norm=norm_img)
        plt.title('Sample {}'.format(idx), fontsize=15, fontweight=FONT_WEIGHT)
        plt.axis('off')
        if (idx+1) % num_samples_show == 0:
            cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
            cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_img, format='%.0e')

    plt.savefig(os.path.join(result_path, 'visualization.png'), bbox_inches='tight', dpi=64)
    plt.close()
    
    
def visualize_samples_deblur(result_path, data, results, log, num_samples, cmap, uq_channel=0, vmin=1e-7):
    for key, value in data.items():
        data[key] = value.squeeze().cpu().detach().numpy() if isinstance(value, torch.Tensor) else value

    psf_clipped = data["psf"].clip(min=vmin)
    error = np.abs(data["gt"] - results['mean'])
    error = error[uq_channel] if data["gt"].shape[0] == 3 else error
    std = results['std'][uq_channel] if data["gt"].shape[0] == 3 else results['std']
    ratio = error / std
    obs_clipped = data['measurement'].clip(min=0.0, max=1.0)
    mean = results['mean'].clip(min=0.0, max=1.0)
    samples = results['samples'].clip(min=0.0, max=1.0)
    
    fig = plt.figure(figsize=(20, 14))
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    norm_psf = LogNorm(vmin=vmin, vmax=psf_clipped.max())
    norm_std = Normalize(vmin=0, vmax=std.max())
    norm_error = Normalize(vmin=0, vmax=error.max())
    norm_ratio = Normalize(vmin=0, vmax=4.0)
    
    ax = plt.subplot(3,4,1)
    if data['measurement'].shape[0] == 3:
        ax.imshow(obs_clipped.transpose(1,2,0))
    else:
        ax.imshow(obs_clipped, cmap='gray', norm=norm_img)
    ax.set_title('Observation', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={psnr(data["gt"], data["measurement"], data_range=1.0):.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'SSIM={ssim(data["gt"], data["measurement"], data_range=1.0):.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.axis('off')
    
    ax = plt.subplot(3,4,5)
    ax.imshow(psf_clipped, cmap='gray', norm=norm_psf)
    ax.set_title('PSF', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_psf)
        
    ax = plt.subplot(3,4,2)
    if mean.shape[0] == 3:
        ax.imshow(mean.transpose(1,2,0))
    else:
        ax.imshow(mean, cmap='gray', norm=norm_img)
        
    ax.set_title(f'Mean $\mu$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={log["psnr"]:.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'SSIM={log["ssim"]:.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'LPIPS={log["lpips"]:.3f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.97, fontsize=15, color='white')
    ax.axis('off')

    ax = plt.subplot(3,4,6)
    if data["gt"].shape[0] == 3:
        ax.imshow(data["gt"].transpose(1,2,0))
    else:
        ax.imshow(data["gt"], cmap='gray', norm=norm_img)
    ax.set_title('Ground Truth $t$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    
    ax = plt.subplot(3,4,3)
    ax.imshow(std, cmap=CMAP_STD, norm=norm_std)
    ax.set_title(f'Std. Dev. $\sigma$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'NLL={log["nll"]:.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.text(s=f'ECE={log["ece"]:.2f}', x=data["gt"].shape[-2]*0.6, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_std)
    
    ax = plt.subplot(3,4,7)
    if data["gt"].shape[0] == 3:
        ax.imshow(std/mean[uq_channel], cmap=CMAP_RATIO)
    else:
        ax.imshow(std/mean, cmap=CMAP_RATIO)
    ax.imshow(std/mean[uq_channel], cmap=CMAP_RATIO)
    ax.set_title(f'$\sigma / \mu$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax)
    
    ax = plt.subplot(3,4,4)
    ax.imshow(error, cmap=CMAP_ERROR, norm=norm_error)
    ax.set_title('Absolute error $|\mu - t|$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_error)
    
    ax = plt.subplot(3,4,8)
    ax.imshow(ratio, cmap=CMAP_RATIO, norm=norm_ratio)
    ax.set_title(f'$|\mu - t| / \sigma$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'OODR={100*log["oodr"]:.2f}%', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_ratio)

    num_samples_show = min(num_samples, 4)
    for idx in range(num_samples_show):
        ax = plt.subplot(3,4,idx+9)
        if samples[idx].shape[0] == 3:
            ax.imshow(samples[idx].transpose(1,2,0))
        else:
            ax.imshow(samples[idx], cmap='gray', norm=norm_img)
        ax.text(s=f'PSNR={psnr(data["gt"], samples[idx], data_range=1.0):.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
        ax.text(s=f'SSIM={ssim(data["gt"], samples[idx], data_range=1.0):.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
        ax.set_title('Sample {}'.format(idx), fontsize=15, fontweight=FONT_WEIGHT)
        ax.axis('off')

    plt.savefig(os.path.join(result_path, 'visualization.png'), bbox_inches='tight', dpi=64)
    plt.close()
    
    
def visualize_samples_fpr(result_path, data, results, log, num_samples, cmap, uq_channel=0, vmin=1e-7):
    for key, value in data.items():
        data[key] = value.squeeze().cpu().detach().numpy() if isinstance(value, torch.Tensor) else value

    error = np.abs(data["gt"] - results['mean'])
    error = error[uq_channel] if data["gt"].shape[0] == 3 else error
    std = results['std'][uq_channel] if data["gt"].shape[0] == 3 else results['std']
    ratio = error / std
    mean = results['mean'].clip(min=0.0, max=1.0)
    samples = results['samples'].clip(min=0.0, max=1.0)

    fig = plt.figure(figsize=(20, 14))
    norm_img = Normalize(vmin=0.0, vmax=1.0)
    norm_amp = LogNorm(vmin=vmin, vmax=data['measurement'].max())
    norm_std = Normalize(vmin=0, vmax=std.max())
    norm_error = Normalize(vmin=0, vmax=error.max())
    norm_ratio = Normalize(vmin=0, vmax=4.0)
    
    ax = plt.subplot(3,4,1)
    ax.imshow(data['measurement'], cmap='gray', norm=norm_amp)
    ax.set_title('Amplitude', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')

    ax = plt.subplot(3,4,2)
    if mean.shape[0] == 3:
        ax.imshow(mean.transpose(1,2,0))
    else:
        ax.imshow(mean, cmap='gray', norm=norm_img)
        
    ax.set_title(f'Mean $\mu$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'PSNR={log["psnr"]:.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='red')
    ax.text(s=f'SSIM={log["ssim"]:.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='red')
    ax.text(s=f'LPIPS={log["lpips"]:.3f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.97, fontsize=15, color='red')
    ax.axis('off')

    ax = plt.subplot(3,4,6)
    if data["gt"].shape[0] == 3:
        ax.imshow(data["gt"].transpose(1,2,0))
    else:
        ax.imshow(data["gt"], cmap='gray', norm=norm_img)
    ax.set_title('Ground Truth $t$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    
    ax = plt.subplot(3,4,3)
    ax.imshow(std, cmap=CMAP_STD, norm=norm_std)
    ax.set_title(f'Std. Dev. $\sigma$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'NLL={log["nll"]:.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='red')
    ax.text(s=f'ECE={log["ece"]:.2f}', x=data["gt"].shape[-2]*0.6, y=data["gt"].shape[-1]*0.08, fontsize=15, color='red')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_std)
    
    ax = plt.subplot(3,4,7)
    if data["gt"].shape[0] == 3:
        ax.imshow(std/mean[uq_channel], cmap=CMAP_RATIO)
    else:
        ax.imshow(std/mean, cmap=CMAP_RATIO)
    ax.imshow(std/mean[uq_channel], cmap=CMAP_RATIO)
    ax.set_title(f'$\sigma / \mu$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax)
    
    ax = plt.subplot(3,4,4)
    ax.imshow(error, cmap=CMAP_ERROR, norm=norm_error)
    ax.set_title('Absolute error $|\mu - t|$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_error)
    
    ax = plt.subplot(3,4,8)
    ax.imshow(ratio, cmap=CMAP_RATIO, norm=norm_ratio)
    ax.set_title(f'$|\mu - t| / \sigma$', fontsize=15, fontweight=FONT_WEIGHT)
    ax.text(s=f'OODR={100*log["oodr"]:.2f}%', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='white')
    ax.axis('off')
    cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
    cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_ratio)

    for idx in range(4):
        ax = plt.subplot(3,4,idx+9)
        if samples[idx].shape[0] == 3:
            ax.imshow(samples[idx].transpose(1,2,0))
        else:
            ax.imshow(samples[idx], cmap='gray', norm=norm_img)
        ax.text(s=f'PSNR={psnr(data["gt"], samples[idx], data_range=1.0):.2f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.08, fontsize=15, color='red')
        ax.text(s=f'SSIM={ssim(data["gt"], samples[idx], data_range=1.0):.3f}', x=data["gt"].shape[-2]*0.53, y=data["gt"].shape[-1]*0.08, fontsize=15, color='red')
        ax.set_title('Sample {}'.format(idx), fontsize=15, fontweight=FONT_WEIGHT)
        ax.axis('off')

    plt.savefig(os.path.join(result_path, 'visualization.png'), bbox_inches='tight', dpi=64)
    plt.close()
    
    
# def visualize_samples_cdp(result_path, samples, mean, std, num_samples, gt, measurement, log, uq_channel=0):

#     error = np.abs(gt - mean)
#     if gt.shape[0] == 3:
#         error = error[uq_channel]
#         std = std[uq_channel]
#     ratio = error / std
#     mean = mean.clip(min=0.0, max=1.0)
#     samples = samples.clip(min=0.0, max=1.0)
    
#     fig = plt.figure(figsize=(20, 14))
#     norm_img = Normalize(vmin=0.0, vmax=1.0)
#     # norm_amp = LogNorm(vmin=1e-4, vmax=amp.max())
#     norm_std = Normalize(vmin=0, vmax=std.max())
#     norm_error = Normalize(vmin=0, vmax=error.max())
#     norm_ratio = Normalize(vmin=0, vmax=4.0)
    
#     ax = plt.subplot(3,4,2)
#     if mean.shape[0] == 3:
#         ax.imshow(mean.transpose(1,2,0))
#     else:
#         ax.imshow(mean, cmap='gray', norm=norm_img)
        
#     ax.set_title(f'Mean $\mu$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
#     ax.text(s=f'PSNR={log["psnr"]:.2f}', x=gt.shape[-2]*0.03, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#     ax.text(s=f'SSIM={log["ssim"]:.3f}', x=gt.shape[-2]*0.53, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#     ax.text(s=f'LPIPS={log["lpips"]:.3f}', x=data["gt"].shape[-2]*0.03, y=data["gt"].shape[-1]*0.97, fontsize=15, color='white')
#     ax.axis('off')

#     ax = plt.subplot(3,4,6)
#     if gt.shape[0] == 3:
#         ax.imshow(gt.transpose(1,2,0))
#     else:
#         ax.imshow(gt, cmap='gray', norm=norm_img)
#     ax.set_title('Ground Truth $t$', fontsize=15, fontweight=FONT_WEIGHT)
#     ax.axis('off')
    
#     ax = plt.subplot(3,4,3)
#     ax.imshow(std, cmap=CMAP_STD, norm=norm_std)
#     ax.set_title(f'Std. Dev. $\sigma$ ({num_samples} samples)', fontsize=15, fontweight=FONT_WEIGHT)
#     ax.text(s=f'NLL={log["nll"]:.2f}', x=gt.shape[-2]*0.03, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#     ax.text(s=f'ECE={log["ece"]:.2f}', x=gt.shape[-2]*0.6, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#     ax.axis('off')
#     cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
#     cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_std)
    
#     ax = plt.subplot(3,4,7)
#     if gt.shape[0] == 3:
#         ax.imshow(std/mean[uq_channel], cmap=CMAP_RATIO)
#     else:
#         ax.imshow(std/mean, cmap=CMAP_RATIO)
#     ax.imshow(std/mean[uq_channel], cmap=CMAP_RATIO)
#     ax.set_title(f'$\sigma / \mu$', fontsize=15, fontweight=FONT_WEIGHT)
#     ax.axis('off')
#     cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
#     cb = plt.colorbar(mappable=ax.images[0], cax=cax)
    
#     ax = plt.subplot(3,4,4)
#     ax.imshow(error, cmap=CMAP_ERROR, norm=norm_error)
#     ax.set_title('Absolute error $|\mu - t|$', fontsize=15, fontweight=FONT_WEIGHT)
#     ax.axis('off')
#     cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
#     cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_error)
    
#     ax = plt.subplot(3,4,8)
#     ax.imshow(ratio, cmap=CMAP_RATIO, norm=norm_ratio)
#     ax.set_title(f'$|\mu - t| / \sigma$', fontsize=15, fontweight=FONT_WEIGHT)
#     ax.text(s=f'OODR={100*log["oodr"]:.2f}%', x=gt.shape[-2]*0.03, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#     ax.axis('off')
#     cax = fig.add_axes([ax.get_position().x1+6e-3, ax.get_position().y0, 1e-2, ax.get_position().height])
#     cb = plt.colorbar(mappable=ax.images[0], cax=cax, norm=norm_ratio)

#     for idx in range(4):
#         ax = plt.subplot(3,4,idx+9)
#         if samples[idx].shape[0] == 3:
#             ax.imshow(samples[idx].transpose(1,2,0))
#         else:
#             ax.imshow(samples[idx], cmap='gray', norm=norm_img)
#         ax.text(s=f'PSNR={psnr(gt, samples[idx], data_range=1.0):.2f}', x=gt.shape[-2]*0.03, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#         ax.text(s=f'SSIM={ssim(gt, samples[idx], data_range=1.0):.3f}', x=gt.shape[-2]*0.53, y=gt.shape[-1]*0.08, fontsize=15, color='white')
#         ax.set_title('Sample {}'.format(idx), fontsize=15, fontweight=FONT_WEIGHT)
#         ax.axis('off')

#     plt.savefig(os.path.join(result_path, 'visualization.png'), bbox_inches='tight', dpi=64)
#     plt.close()
    

def tsne(
    images: np.ndarray, 
    n_components: int=2, 
    pca_components: int=16, 
    perplexity: int=16, 
    max_iter: int=1000
):
    # Reduce the dimensionality using PCA first (optional, but often helpful).
    flattened_samples = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=pca_components) 
    pca_features = pca.fit_transform(flattened_samples)
    
    # Perform t-SNE on the PCA features
    tsne = TSNE(n_components=n_components, random_state=777, perplexity=perplexity, max_iter=max_iter)
    
    return tsne.fit_transform(pca_features)
