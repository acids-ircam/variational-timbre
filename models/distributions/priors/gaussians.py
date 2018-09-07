# -*- coding: utf-8 -*-

import torch.distributions as dist
from numpy import zeros, ones, ndarray
from torch import from_numpy, Tensor, index_select, LongTensor, cat
from torch.autograd import Variable
from utils.onehot import fromOneHot
import random

class Prior(object):
    def __init__(self, *args, **kwargs):
        super(Prior, self).__init__()
        self.dim = 0
        self.dist = dist.Distribution
        self.params = ()
        
    def __call__(self, cuda=False, *args, **kwargs):
        draw = self.dist.sample(*self.params)
        if cuda:
            draw = draw.cuda()
        return draw
    
    def get_params(self, device='cpu', *args, **kwargs):
        params = [ p.to(device) for p in self.params ]
        return tuple(params)

class IsotropicGaussian(Prior):
    def __init__(self, dim, *args, **kwargs):
        super(IsotropicGaussian, self).__init__()
        self.params = (from_numpy(zeros((1, dim), 'float32')),
                       from_numpy(ones((1, dim), 'float32')))
        self.params[0].requires_grad_(False)
        self.params[1].requires_grad_(False)
        self.dist = dist.Normal


class DiagonalGaussian(Prior):
    def __init__(self, params):
        assert len(params)==2
        self.dim = params[0].size(1)
        self.dist = dist.Normal
        self.params = params
        
        
class MultiGaussian(Prior):
    def __init__(self, params, dist=dist.Normal):
        self.dim = params[0].size(1); self.dist = dist
        self.params = []
        for i in range(len(params)):
            p = params[i]
            if issubclass(type(p), ndarray):
                p = from_numpy(p)
            if issubclass(type(p), Tensor):
                p = Variable(p)
            self.params.append(p)
        self.params = tuple(self.params) # Warning! Here params are Gaussian Parameters for each class
        
    def remove_undeterminate(self, y, undeterminate_id=-1):
        for i in range(y.shape[0]):
            if y.data[i, -1] != 0:
                random_cat = random.randrange(0, y.shape[1]-1)
                y[i, random_cat] = 1
        y = y[:, :-1]
        return y
    
    def __call__(self, y=[], cuda=False, *args, **kwargs):
        with_undeterminate = kwargs.get('with_undeterminate', False)
        if with_undeterminate:
            undeterminate_id = kwargs.get('undeterminate_id', -1)
            y = self.remove_undeterminate(y, undeterminate_id)
        y = fromOneHot(y)
        if y.is_cuda:
            y = y.cpu()
        z = Variable(Tensor(len(y), self.dim))
        for i in range(len(y)):
            param = []
            for p in range(len(self.params)):
                p = self.params[p][y[i]]
                if not issubclass(type(p), Variable):
                    p = Variable(p, retain_graph=False)
                param.append(p)
            param = tuple(param)
            z[i, :] = self.dist(*param)
            if cuda:
                z = z.cuda()
        return z
    
    
    def get_params(self, y=[], cuda=False, *args, **kwargs):
        params = []
        y = fromOneHot(y)
        if cuda:
            y = y.cuda()
        for i in range(len(self.params)):
            if cuda:
                param = self.params[i].cuda()
            else:
                param = self.params[i]
            p = index_select(param, 0, y) 
            params.append(p)
        return tuple(params)
        
