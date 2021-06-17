"""
Various helper network modules
"""
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """ a simple MLP"""

    def __init__(self, in_dim, sizes, out_dim, non_linearity):
        super().__init__()
        self.non_linearity = non_linearity
        self.in_layer = nn.Linear(in_dim, sizes[0])
        self.out_layer = nn.Linear(sizes[-1], out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)])

    def forward(self, x):
        x = self.non_linearity(self.in_layer(x))
        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                x = self.non_linearity(layer(x))
        x = self.out_layer(x)
        return x

class MLPR(nn.Module):
    def __init__(self, in_dim, sizes, out_dim, non_linearity, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.sizes = sizes
        self.out_dim = out_dim
        self.non_linearity = non_linearity
        self.residual = residual
        self.in_layer = nn.Linear(in_dim, self.sizes[0])
        self.out_layer = nn.Linear(self.sizes[-1], out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)])

    def forward(self, x):
        x = self.non_linearity(self.in_layer(x))

        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                residual = x
                x = self.non_linearity(layer(x))
            else:
                x = self.non_linearity(residual + layer(x))

        x = self.out_layer(x)
        return x
