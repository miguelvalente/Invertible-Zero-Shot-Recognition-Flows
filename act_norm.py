import einops
import torch
import torch.nn as nn

from transform import Transform

#  Code adapted from : https://github.com/didriknielsen/survae_flows/

class _ActNormBijection(Transform):
    '''
    Base class for activation normalization [1].
    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def __init__(self, num_features, data_dep_init=True, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.data_dep_init = data_dep_init
        self.eps = eps

        self.register_buffer('initialized', torch.zeros(1) if data_dep_init else torch.ones(1))
        self.register_params()

    def data_init(self, x):
        self.initialized += 1.
        with torch.no_grad():
            x_mean, x_std = self.compute_stats(x)
            x_mean.type(x.type())
            x_std.type(x.type())
            self.shift = nn.Parameter(x_mean)

            self.log_scale = nn.Parameter(torch.log(x_std + self.eps))

    def forward(self, x, context=None):
        if self.training and not self.initialized:
            self.data_init(x)
        z = (x - self.shift) * torch.exp(-self.log_scale)
        ldj = torch.sum(-self.log_scale).expand(x.shape[:-1]) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z, context=None):
        return self.shift + z * torch.exp(self.log_scale)

    def register_params(self):
        '''Register parameters shift and log_scale'''
        raise NotImplementedError()

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        raise NotImplementedError()

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        raise NotImplementedError()

class ActNormBijection(_ActNormBijection):
    '''
    Activation normalization [1] for inputs on the form (B,D).
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def register_params(self):
        '''Register parameters shift and log_scale'''
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, self.num_features)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features)))

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True)
        return x_mean, x_std

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return 1
