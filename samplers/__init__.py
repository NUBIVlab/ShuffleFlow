from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, img_shape, net, forward_op, device='cuda', **kwargs):
        self.img_shape = img_shape
        self.net = net
        self.forward_op = forward_op
        self.device = device
    
    @abstractmethod
    def __call__(self, observation, num_samples=1, **kwargs):
        '''
        Args:
            - observation: observation for one single ground truth
            - num_samples: number of samples to generate for each observation
        '''
        pass
    
from .daps import DAPS
from .diffusion import *
from .dps import DPS
from .pnp_dm import PnPDM
from .score_ald import ScoreALD
from .score_mri import ScoreMRI
