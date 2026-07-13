import numpy as np
import torch
from hydra.utils import instantiate
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.dataio import load_results_ps, load_results_vi
from utils.name import *


def psnr(x, gt, data_range=None):
    if data_range is None:
        data_range = gt.max()-gt.min()
    return float(peak_signal_noise_ratio(x, gt, data_range=data_range))


def ssim(x, gt, data_range=None):
    if data_range is None:
        data_range = gt.max() - gt.min()
    if len(x.shape) == 3:
        return float(structural_similarity(x, gt, channel_axis=0, data_range=data_range))
    else:
        return float(structural_similarity(x, gt, data_range=data_range))


def nll(x, mean, std):
    nll = (x - mean) ** 2 / (2 * std ** 2) + 0.5 * torch.log(2 * torch.pi * std ** 2)
    return nll.mean()


def oodr(x, mean, std, threshold=3.0):
    ratio = torch.abs(mean-x) / std
    return (ratio > threshold).float().mean()


def ece(samples, gt, percent_diff=0.05, device=None):
    """Expected Calibration Error. Uses the same device as samples when possible for speed."""
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
    if isinstance(gt, np.ndarray):
        gt_t = torch.from_numpy(gt)
    else:
        gt_t = gt

    # Run on GPU if available and device not forced to CPU (much faster for large tensors)
    if device is None and samples.is_cuda:
        device = samples.device
    elif device is None:
        device = torch.device('cpu')
    samples = samples.to(device)
    gt_t = gt_t.to(device)

    percentages = [round(percent, 2) for percent in np.arange(0, 1, percent_diff)]

    # Find optimal delta value. Delta is a small positive number used to widen predictive intervals
    # In order to account for 0 pixels (unattainable by Softmax-outputted models, like UncertaINR)
    # At the cost of slightly wider predictions. We search for delta over a grid of values.
    deltas = np.append(np.logspace(-20, -1, 21), 0)
    best_delta = 0
    best_ece = np.inf
    for delta in deltas:
        ece_val = 0.0
        for percent in percentages:
            lb = torch.quantile(samples, 0.5 - percent / 2, dim=0, keepdim=True)
            ub = torch.quantile(samples, 0.5 + percent / 2, dim=0, keepdim=True)
            mean_pt = ((gt_t > lb - delta) & (gt_t < ub + delta)).float().mean()
            ece_val += abs(mean_pt.item() - percent) * percent_diff
        if ece_val < best_ece:
            best_delta = delta
            best_ece = ece_val

    # Using best_delta to calulcate and save final values for coverage and ECE。
    ece_val = 0.0
    for percent in percentages:
        lb = torch.quantile(samples, 0.5 - percent / 2, dim=0, keepdim=True)
        ub = torch.quantile(samples, 0.5 + percent / 2, dim=0, keepdim=True)
        mean_pt = ((gt_t > lb - best_delta) & (gt_t < ub + best_delta)).float().mean()
        ece_val += abs(mean_pt.item() - percent) * percent_diff

    return ece_val


def print_metrics(config):
    psnr_list, ssim_list, lpips_list, nll_list, ece_list, oodr_list, time_list, vram_list = [], [], [], [], [], [], [], []
    
    name_data = get_name_data(config.inverse_problem.dataset)
    name_operator = get_name_operator(config.inverse_problem.operator)
    testset = instantiate(config.inverse_problem.dataset, name_data=name_data, name_operator=name_operator, _recursive_=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    for idx, data in enumerate(test_loader):
        data_idx = testset.idx_map(idx)
        if 'sampler' in config:
            results, log = load_results_ps(config, data_idx)
        elif 'flow' in config:
            results, log = load_results_vi(config, data_idx)
        else:
            raise ValueError(f'Unknown method!')
        
        psnr_list.append(log['psnr'])
        ssim_list.append(log['ssim'])
        lpips_list.append(log['lpips'])
        nll_list.append(log['nll'])
        ece_list.append(log['ece'])
        oodr_list.append(log['oodr']*100)
        time_list.append(log['time']/60)
        vram_list.append(log['max_vram'])

    print(f'PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f}')
    print(f'SSIM: {np.mean(ssim_list):.3f} ± {np.std(ssim_list):.3f}')
    print(f'LPIPS: {np.mean(lpips_list):.3f} ± {np.std(lpips_list):.3f}')
    print(f'NLL: {np.mean(nll_list):.2f} ± {np.std(nll_list):.2f}')
    print(f'ECE: {np.mean(ece_list):.3f} ± {np.std(ece_list):.3f}')
    print(f'OODR: {np.mean(oodr_list):.2f} ± {np.std(oodr_list):.2f} %')
    print(f'Time: {np.mean(time_list):.1f} ± {np.std(time_list):.1f} min')
    print(f'VRAM: {np.mean(vram_list):.1f} ± {np.std(vram_list):.1f} GB')
    
    
    