import math

import torch
from torch import nn


def get_N_freq_nyquist(samples):
    nyquist_rate = 1 / (2 * (2 * 1 / samples))
    return int(math.floor(math.log(nyquist_rate, 2)))


class PosEncodingNeRF(nn.Module):
    def __init__(self, in_features, n_freqs=4, sidelength=None, use_nyquist=False):
        super().__init__()

        self.in_features = in_features

        # if self.in_features == 3:
        #     self.N_freq = 10
        # elif self.in_features <= 2:
        #     assert sidelength is not None
        #     if isinstance(sidelength, int):
        #         sidelength = (sidelength, sidelength)
        #     self.N_freq = 4
        #     if use_nyquist:
        #         self.N_freq = get_N_freq_nyquist(min(sidelength[0], sidelength[1]))
        self.n_freqs = n_freqs
        self.out_dim = in_features + 2 * in_features * self.n_freqs

    def forward(self, coords):
        # print(coords.shape)
        coords = coords.view(coords.shape[0], -1, self.in_features)
        # print(coords.shape)
        coords_encoded = coords
        for i in range(self.n_freqs):
            for j in range(self.in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * torch.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * torch.pi * c), -1)
                coords_encoded = torch.cat((coords_encoded, sin, cos), axis=-1)
        # print(coords_encoded.shape)
        return coords_encoded.reshape(coords.shape[0], -1, self.out_dim)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-math.sqrt(6 / num_input) / 30, math.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            
def kaiming_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
           
            
class Sine(nn.Module):
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega
        
    def forward(self, x):
        return torch.sin(self.omega * x)
    
       
class SIREN(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, act_func='sin', pos_encoding=False, n_freqs=0):
        super().__init__()
        
        if act_func == 'sin':
            self.act_func = Sine()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'softplus':
            self.act_func = nn.Softplus()
        else:
            raise ValueError('Activation function {act_func} not supported.')
        
        if pos_encoding:
            self.pos_encoding = PosEncodingNeRF(in_features, n_freqs=n_freqs)
            in_features = self.pos_encoding.out_dim
        else:
            self.pos_encoding = False
        
        if hidden_layers == -1:
            self.mlp = None
        else:
            layers = [nn.Linear(in_features, hidden_features), self.act_func]
            for _ in range(hidden_layers):
                layers += [nn.Linear(hidden_features, hidden_features), self.act_func]
            layers.append(nn.Linear(hidden_features, out_features))
            self.mlp = nn.Sequential(*layers)
        
            # Initialize weights.
            if act_func == 'sin':
                self.mlp.apply(sine_init)
                self.mlp[0].apply(first_layer_sine_init)
            elif act_func in ['relu', 'sigmoid']:
                self.mlp.apply(kaiming_init)
            else:
                self.mlp.apply(nn.init.xavier_uniform_)
            
    def forward(self, x):
        coords = self.pos_encoding(x) if self.pos_encoding else x
        return self.mlp(coords) if self.mlp is not None else coords
    