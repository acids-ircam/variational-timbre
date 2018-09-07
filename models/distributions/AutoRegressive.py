# -*- coding: utf-8 -*-

from .Spectral import Spectral
import torch.distributions as dist

class AutoRegressiveNormal(dist.Normal):
    pass

class AutoRegressiveBernoulli(dist.Bernoulli):
    pass

def AutoRegressive(distribution = dist.normal):
    if distribution == dist.Normal or distribution == Spectral:
        return AutoRegressiveNormal
    elif distribution == dist.Bernoulli:
        return AutoRegressiveBernoulli
