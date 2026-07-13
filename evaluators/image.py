import logging
from typing import Dict

import torch
from piq import LPIPS, psnr, ssim

from utils.metrics import ece, nll, oodr

from . import BaseEvaluator


class ImageEvaluator(BaseEvaluator):
    def __init__(self, forward_operator=None):
        self.eval_batch = 32
        self.lpips = LPIPS(replace_pooling=True, reduction='none').eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        metric_list = {
            'psnr': lambda pred, std, samples, gt: psnr(pred.clip(0, 1), gt.clip(0, 1), data_range=1.0, reduction='none'),
            'ssim': lambda pred, std, samples, gt: ssim(pred.clip(0, 1), gt.clip(0, 1), data_range=1.0, reduction='none'),
            'lpips': lambda pred, std, samples, gt: self.lpips(pred.clip(0, 1), gt.clip(0, 1)),
            'nll': lambda pred, std, samples, gt: nll(gt, pred, std),
            'ece': lambda pred, std, samples, gt: ece(samples, gt),
            'oodr': lambda pred, std, samples, gt: oodr(gt, pred, std),
        }
        super(ImageEvaluator, self).__init__(metric_list, forward_operator=forward_operator)

    def __call__(self, samples: torch.Tensor, data: Dict, time: float=None, max_vram: float=None):
        mean = samples.mean(0, keepdim=True)
        std = samples.std(0, keepdim=True)
        
        metric_dict = {'time': time, 'max_vram': max_vram}
        self.metric_state['time'].append(time)
        self.metric_state['max_vram'].append(max_vram)
        
        if self.data_fit_loss:
            data_fit_loss = self.eval_data_fitting_loss(mean, data['measurement']).item()
            metric_dict['data_fit_loss'] = data_fit_loss
            self.metric_state['data_fit_loss'].append(data_fit_loss)
            
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            val = metric_func(mean, std, samples, data['gt']).item()
            metric_dict[metric_name] = val
            self.metric_state[metric_name].append(val)
        return metric_dict
    
    @staticmethod
    def print_metrics(metric_dict: Dict, title: str='', logger: logging.Logger=None):
        if logger is not None:
            logger.info(title.center(60, '-'))
            logger.info(f'PSNR: {metric_dict["psnr"]:.2f}\t   SSIM: {metric_dict["ssim"]:.3f}\t   LPIPS: {metric_dict["lpips"]:.3f}')
            logger.info(f'NLL: {metric_dict["nll"]:.2f}\t   ECE: {metric_dict["ece"]:.2f}\t   OODR: {100*metric_dict["oodr"]:.2f}%')
            logger.info(f'Time: {metric_dict["time"]:.1f} s\t  Peak GPU memory: {metric_dict["max_vram"]:.2f} GB')
            logger.info(''.center(60, '-'))
        else:
            print(title.center(60, '-'))
            print(f'PSNR: {metric_dict["psnr"]:.2f}\t   SSIM: {metric_dict["ssim"]:.3f}\t   LPIPS: {metric_dict["lpips"]:.3f}')
            print(f'NLL: {metric_dict["nll"]:.2f}\t   ECE: {metric_dict["ece"]:.2f}\t   OODR: {100*metric_dict["oodr"]:.2f}%')
            print(f'Time: {metric_dict["time"]:.1f} s\t  Peak GPU memory: {metric_dict["max_vram"]:.2f} GB')
            print(''.center(60, '-'))