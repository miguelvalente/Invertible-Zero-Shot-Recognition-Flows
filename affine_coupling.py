
import einops
import torch
from torch import nn
from torch.utils import checkpoint

from nets import MLP, MLPR
from transform import Transform


class AffineCoupling(Transform):
    def __init__(self, input_dim, hidden_dims, non_linearity, net=MLP, affine_scale_eps=2, affine_eps=0.1, context_dim=0, event_dim=-1):
        super().__init__()
        self.event_dim = event_dim
        self.input_dim = input_dim
        self.split_dim = input_dim // 2
        self.context_dim = context_dim
        out_dim = (self.input_dim - self.split_dim) * 2
        if net == 'MLP':
            self.nn_s = MLP(self.split_dim + context_dim, hidden_dims, out_dim, non_linearity)
            self.nn_s = MLP(self.split_dim + context_dim, hidden_dims, out_dim, non_linearity)
            self.nn = MLP(self.split_dim + context_dim, hidden_dims, out_dim, non_linearity)
        elif net == 'MLPR':
            self.nn_s = MLPR(self.split_dim + context_dim, hidden_dims, self.split_dim, non_linearity)
            self.nn_t = MLPR(self.split_dim + context_dim, hidden_dims, self.split_dim, non_linearity)

        self.affine_eps = affine_eps
        self.affine_scale_eps = affine_scale_eps

    def forward(self, x, context=None):
        x2_size = self.input_dim - self.split_dim
        x1, x2 = x.split([self.split_dim, x2_size], dim=self.event_dim)

        shift, scale = self.nn(x1).split([x2_size, x2_size], dim=-1)
        scale = torch.sigmoid(scale + self.affine_scale_eps) + self.affine_eps

        y1 = x1

        y2 = x2 + shift
        y2 = y2 * scale

        ldj = torch.einsum('bn->b', torch.log(scale))
        return torch.cat([y1, y2], dim=self.event_dim), ldj

    def inverse(self, y, context=None):
        y2_size = self.input_dim - self.split_dim
        y1, y2 = y.split([self.split_dim, y2_size], dim=self.event_dim)

        shift, scale = self.nn(y1).split([y2_size, y2_size], dim=-1)
        scale = torch.sigmoid(scale + self.affine_scale_eps) + self.affine_eps

        x1 = y1

        x2 = y2 / scale
        x2 = x2 - shift

        return torch.cat([x1, x2], dim=self.event_dim)
