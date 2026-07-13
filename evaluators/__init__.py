from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch

from forward_operators import BaseOperator


class BaseEvaluator(ABC):
    def __init__(self, metric_list: List, forward_operator: BaseOperator=None, data_fit_loss=False):
        self.metric_list = metric_list
        self.forward_operator = forward_operator
        self.data_fit_loss = data_fit_loss
        if data_fit_loss:
            assert forward_operator is not None, "forward_operator must be provided for data fitting loss evaluation"
        
        self.device = forward_operator.device if forward_operator is not None else 'cpu'
        self.metric_state = {key: [] for key in metric_list.keys()}
        if data_fit_loss:
            self.metric_state['data_fit_loss'] = []
        self.metric_state['time'] = []
        self.metric_state['max_vram'] = []

    def eval_data_fitting_loss(self, pred: torch.Tensor, measurement: torch.Tensor):
        """
        Args:
            - pred (torch.Tensor): (N, C, H, W) unnormalized
            - measurement (torch.Tensor): (N, C, H, W)
        Returns:
            - data_misfit (torch.Tensor): (N,), data misfit
        """
        data_fit_loss = self.forward_operator.loss(pred, measurement, unnormalize=False)
        return torch.sqrt(data_fit_loss)

    @abstractmethod
    def __call__(self, pred: torch.Tensor, ground_truth: torch.Tensor, measurement: torch.Tensor=None, time: float=None, vram: float=None) -> Dict:
        pass

    def aggregate(self):
        metric_aggregated = {}
        for key, val in self.metric_state.items():
            metric_aggregated[key] = np.mean(val)
            metric_aggregated[f'{key}_std'] = np.std(val)
        return metric_aggregated
