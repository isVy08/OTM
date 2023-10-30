import torch
import numpy as np 
import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm

class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU(), bias=True, p_drop = 0.0):
        super(MLP, self).__init__()
        layers = []
        for l in range(len(dims) - 1):
            layers.append(nn.Linear(dims[l], dims[l+1], bias=bias))
            layers.append(act)
            if p_drop > 0.0:
                layers.append(nn.Dropout(p=p_drop))

        layers.append(nn.Linear(dims[l+1], dims[l+1], bias=bias))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class SpectralLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))
        # self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))
        # self.spectral_linear(torch.Tensor(torch.ones(1, input_dim)))
        # remove_spectral_norm(self.spectral_linear)
        # self.spectral_linear.weight = nn.Parameter(self.spectral_linear.weight * self.k_lipschitz)

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_linear(x)
        # y = self.spectral_linear(x)
        return y

def linear_sequential(input_dims, hidden_dims, output_dim, act, k_lipschitz=None, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(act)
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)

class CNN(nn.Module):
    def __init__(self, dims, act=nn.ReLU(), p_drop = 0.0):
        super().__init__()

        layers = []
        for l in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[l], dims[l+1], kernel_size = 2, stride = 2))
            layers.append(act)
            if p_drop > 0.0:
                layers.append(nn.Dropout(p=p_drop))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, pooling = None):
        x = self.model(x)
        kernel = x.size(-1)
        if pooling == 'average':
            x = nn.functional.avg_pool1d(x, kernel)
            x = x.squeeze(-1)
        elif pooling == 'max':
            x = nn.functional.max_pool1d(x, kernel)
            x = x.squeeze(-1)
        return x


class Discriminator(nn.Module):
    def __init__(self, D, lr = 1e-4):
        super(Discriminator, self).__init__()
        self.D = D
        H = self.D * 2
        self.module = linear_sequential(self.D, [H, H], self.D, nn.ReLU(), 1.0, 0.2)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, betas=(0.5, 0.999))

    def forward(self, x):
        x = self.module(x)
        return x
