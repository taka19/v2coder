from dataclasses import dataclass
import math
import torch
from torch import Tensor

LOG2PI = 1.8378770664093453

def _num_channels(shape, dim):
    if dim is None:
        return 1
    if isinstance(dim, int):
        return shape[dim]
    return math.prod(shape[d] for d in dim)

def _sum(x, dim):
    if dim is None:
        return x
    return x.sum(dim)

@dataclass()
class GaussParam:
    mu: Tensor
    log_lambda: Tensor

    def apply(self, func):
        return GaussParam(func(self.mu), func(self.log_lambda))

def negative_log_prob1d(x: Tensor, dist: GaussParam, dim: int=None) -> Tensor:
    shape = torch.broadcast_shapes(x.size(), dist.mu.size(), dist.log_lambda.size())
    C = _num_channels(shape, dim)
    return 0.5 * (
        LOG2PI * C - _sum(dist.log_lambda, dim)
        + _sum(torch.square((x - dist.mu) * torch.exp(0.5 * dist.log_lambda)), dim=dim)
    )

def kld1d(q: GaussParam, p: GaussParam, dim: int=None) -> Tensor:
    shape = torch.broadcast_shapes(q.mu.size(), q.log_lambda.size(), p.mu.size(), p.log_lambda.size())
    C = _num_channels(shape, dim)
    return 0.5 * (
        - C + _sum(q.log_lambda, dim=dim) - _sum(p.log_lambda, dim=dim)
        + _sum(torch.square((q.mu - p.mu) * torch.exp(0.5 * p.log_lambda)), dim=dim)
        + _sum(torch.exp(p.log_lambda - q.log_lambda), dim=dim)
    )

def multiply(r: GaussParam, p: GaussParam) -> GaussParam:
    log_lambda = torch.logaddexp(r.log_lambda, p.log_lambda)
    mu = (
        torch.exp(r.log_lambda - log_lambda) * r.mu
        + torch.exp(p.log_lambda - log_lambda) * p.mu
    )
    return GaussParam(mu, log_lambda)

def sample(dist: GaussParam, eps: Tensor=None) -> Tensor:
    std = torch.exp(-0.5 * dist.log_lambda)
    if eps is None: eps = torch.randn_like(std)
    x = dist.mu + eps * std
    return x
