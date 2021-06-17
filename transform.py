import torch
import torch.nn as nn
import torch.nn.functional as F

class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None):
        """
        Forward transform.
        Computes `z <- x` and the log-likelihood contribution term `log C`
        such that `log p(x) = log p(z) + log C`.
        Args:
            x: Tensor, shape (batch_size, ...)
        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z, context=None):
        """
        Inverse transform.
        Computes `x <- z`.
        Args:
            z: Tensor, shape (batch_size, ...)
        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()


class PreConditionApplier(Transform):
    def __init__(self, transform, pre_conditioner):
        super().__init__()
        self.pre_conditioner = pre_conditioner
        self.transform = transform

    def forward(self, x, context=None):
        context_for_transform = self.pre_conditioner(x, context)
        x, ldj = self.transform(x, context=context_for_transform)
        return x, ldj

    def inverse(self, y, context):
        context_for_transform = self.pre_conditioner(y, context)
        y = self.transform.inverse(y, context=context_for_transform)
        return y


class Flow(Transform):
    '''Wrapper for merging multiple transforms'''

    def __init__(self, transform_list, base_dist, sample_dist=None):
        super().__init__()
        self.base_dist = base_dist
        self.sample_dist = sample_dist if sample_dist is not None else base_dist

        self.transforms = nn.ModuleList(transform_list)

    def log_prob(self, x, context=None):
        log_prob = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for index, transform in enumerate(self.transforms):
            x, ldj = transform(x, context=None)
            log_prob += ldj
            if x.isnan().any() or x.isinf().any():
                Exception("Nan or Inf")
        log_prob += self.base_dist.log_prob(x, context=context)
        return log_prob

    def generation(self, z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def sample(self, num_samples, n_points, context=None):
        z = self.sample_dist.sample(num_samples, n_points=n_points)
        for transform in reversed(self.transforms):
            z = transform.inverse(z, context=context)
        return z

    def centralizing_loss(self, data, targets, cs):
        """
        Centralizing loss
        """
        centralizing_loss = 0.0
        means = torch.stack([data[targets == t].mean(axis=0) for t in targets.unique()])
        gens = self.generation(F.pad(cs[targets.unique()], (0, 2)))

        return torch.norm(gens - means, dim=1).sum()

    def mmd_loss(self, data, cu, k=0):
        """
        Calculates MMD respective to a kernel function.
        k==0: Inverse MultiQuadratic Kernel
        """
        mmd_loss = 0.0
        n = len(data)

        v_hat = self.generation(torch.cat((cu.repeat(n, 1),
                                           self.base_dist.visual_distribution.sample([n])), dim=1))

        if k == 0:
            mmd_loss = ((2 / (n**2)) * self.inverse_quadratic(data, v_hat).sum() -
                        (1 / (n * (n - 1))) * (self.inverse_quadratic(data, data) +
                                               self.inverse_quadratic(v_hat, v_hat)).fill_diagonal_(0).sum())

        return mmd_loss

    def inverse_quadratic(self, data1, data2):
        return (2 * data1.shape[1]) / (2 * data1.shape[1] + torch.cdist(data1, data2).square())

class IdentityTransform(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None):
        return x, 0

    def inverse(self, y, context=None):
        return y
