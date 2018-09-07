# -*- coding: utf-8 -*-

import torch.distributions as dist
from .prior_prior import Prior, ClassPrior
from numpy import ones, ndarray
from torch import from_numpy, Tensor, index_select, LongTensor, cat, zeros, ones
from torch.autograd import Variable
from utils.onehot import fromOneHot
import random


class IsotropicGaussian(Prior):
    def __init__(self, dim, *args, **kwargs):
        super(IsotropicGaussian, self).__init__()
        self.params = (zeros((1, dim)),
                       ones((1, dim)))
        self.params[0].requires_grad_(False)
        self.params[1].requires_grad_(False)
        self.dist = dist.Normal


class DiagonalGaussian(Prior):
    def __init__(self, params):
        assert len(params)==2
        self.dim = params[0].size(1)
        self.dist = dist.Normal
        self.params = params
        
        
class ClassGaussian(ClassPrior, DiagonalGaussian):
    def __init__(self, params):
        ClassPrior.__init__(self, params, dist.Normal)