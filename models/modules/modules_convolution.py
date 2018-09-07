#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:44:23 2018

@author: chemla
"""

from models.modules.modules_bottleneck import MLPLayer, MLP
#import torch
#import torch.nn as nn
from functools import reduce
from collections import OrderedDict
import torch
import torch.nn as nn
import pdb
import torch.distributions as dist
import numpy as np

class ConvLayer(nn.Module):
    conv_modules = {1: torch.nn.Conv1d, 2:torch.nn.Conv2d, 3:torch.nn.Conv3d}
    dropout_modules = {1: nn.Dropout, 2:nn.Dropout2d, 3:nn.Dropout3d}
    bn_modules = {1: nn.BatchNorm1d, 2:nn.BatchNorm2d, 3:nn.BatchNorm3d}
    pool_modules = {1:nn.MaxPool1d, 2:nn.MaxPool2d, 3:nn.MaxPool3d}
    def __init__(self, in_channels, out_channels, dim, kernel_size, pool=None, dropout=0.5, padding=0, nn_lin="ReLU", batch_norm=True, return_indices=False):
        super(ConvLayer, self).__init__()
        self.return_indices = return_indices
        self.pool_indices = None
        self.add_module('conv_module', self.conv_modules[dim](in_channels, out_channels, kernel_size, padding=padding))
        if not dropout is None:
            self.dropout = True
            self.add_module('dropout_module', self.dropout_modules[dim](0.5))
        else:
            self.dropout = False
        self.add_module('batch_norm', self.bn_modules[dim](out_channels))
        if not 'nn_lin' is None:
            self.add_module('nn_lin', getattr(nn, nn_lin)())
        if not pool is None:
            self.pooling = True
            self.init_pooling(dim, pool, return_indices)
        else:
            self.pooling = False
        
    def init_pooling(self, dim, kernel_size, return_indices):
        self.add_module('pool_module', self.pool_modules[dim](kernel_size, return_indices=self.return_indices))
        
    def forward(self, x):
        current_out = self._modules['conv_module'](x)
        if self.dropout:
            current_out = self._modules['dropout_module'](current_out)
        current_out = self._modules['batch_norm'](current_out)
        current_out = self._modules['nn_lin'](current_out)
        pool_indices = None
        if self.pooling:
            current_out = self._modules['pool_module'](current_out)
            try:
                current_out, pool_indices = current_out
                pool_indices.requires_grad_(False)
            except Exception as e:
                raise Warning(str(e))
                pass
        if not pool_indices is None:
            self.pool_indices = pool_indices
        return current_out
    
    def get_pooling_indices(self):
        return self.pool_indices
        
class DeconvLayer(ConvLayer):
    pool_modules = {1:nn.MaxUnpool1d, 2:nn.MaxUnpool2d, 3:nn.MaxUnpool3d}
    def __call__(self, x, indices=None):
        if self.pooling:
            x = self._modules['pool_module'](x, indices)

        current_out = self._modules['conv_module'](x)
        if self.dropout:
            current_out = self._modules['dropout_module'](current_out)
        current_out = self._modules['batch_norm'](current_out)
        current_out = self._modules['nn_lin'](current_out)
        return current_out
    
    def init_pooling(self, dim, kernel_size, *args, **kwargs):
        self.add_module('pool_module', self.pool_modules[dim](kernel_size))
    
    
#def multi_tensor
    
class Convolutional(nn.Module):
    conv_module = ConvLayer
    def __init__(self, pins, pouts, phidden={"input_channels":1, "dim":[1,64,32,8], "paddings":[0,0,0,0], "conv_dim":2, "kernel_size":[5,5,5]}, nn_lin="ReLU", name="", return_indices=True, *args, **kwargs):
        super(Convolutional, self).__init__()
        self.pins = pins
        self.pouts = pouts
        self.dim = phidden.get('conv_dim', 2)
        self.n_channels = phidden["dim"]
        self.depth =len(phidden['kernel_size'])
        self.kernel_sizes = phidden["kernel_size"]
        self.pool = phidden.get('pool', [None]*self.depth)
        self.paddings = phidden.get('paddings', [0]*self.depth)
        self.input_channels = phidden.get('input_channels', 1)
        self.return_indices = return_indices
        if not issubclass(type(nn_lin), list):
            self.nn_lin = [nn_lin]*self.depth
        else:
            assert len(nn_lin) == self.depth
            self.nn_lin = nn_lin
        
        for k in self.kernel_sizes:
            assert k % 2 == 1
        assert len(self.n_channels)==len(self.kernel_sizes)+1   
        
        conv_modules = []
        for l in range(self.depth):
            conv_modules.append(self.conv_module(self.n_channels[l], self.n_channels[l+1],  self.dim, self.kernel_sizes[l], self.pool[l], padding=self.paddings[l], nn_lin=self.nn_lin[l], return_indices = return_indices, *args, **kwargs))
        self.conv_encoders = nn.ModuleList(conv_modules)
            

    def forward(self, x):
        if issubclass(type(x), list):
            x = torch.cat(x, )
        data_dim = np.array(x.shape[1:])
        out = x.reshape(x.size(0), 1, *tuple(data_dim))
        # successively pass into convolutional modules
        for l in range(self.depth):
            out = self.conv_encoders[l](out)
        return out
    
    def get_output_conv_length(self):
        current_length = self.pins['dim']
        if not issubclass(type(current_length), list):
            current_length = [current_length]
        current_output = np.array(current_length)

        for l in range(self.depth):
            # conv layers
            current_module = self.conv_encoders[l]._modules['conv_module']
            current_output = current_output + 2*np.array(current_module.padding) - np.array(current_module.dilation) * (np.array(current_module.kernel_size)-1) - 1
            current_output = np.floor(current_output/np.array(current_module.stride) + 1)
            # pooling layers
            if not self.pool[l] is None:
                current_module = self.conv_encoders[l]._modules['pool_module']
                current_output = current_output + 2*np.array(current_module.padding) - np.array(current_module.dilation) * (np.array(current_module.kernel_size)-1) - 1
                current_output = np.floor(current_output/np.array(current_module.stride) + 1)
        return current_output
        
    def get_pooling_indices(self):
        pooling_indices = []
        for l in range(self.depth):
            pooling_indices.append(self.conv_encoders[l].get_pooling_indices())
        return pooling_indices
    
class ConvolutionalLatent(Convolutional):
    def __init__(self, pins, pouts, phidden={"dim":[64,32,8], "conv_dim":2, "kernel_size":[16,8,4]}, *args, **kwargs):
        # input parameters
        assert not issubclass(type(pins), list) or len(pins)==1
        if issubclass(type(pins), list):
            pins = dict(pins[0])
        # hidden parameters
        phidden = dict(phidden)
        phidden['dim'] = [1]+phidden['dim']
        
        paddings = []
        for k in phidden['kernel_size']:
            paddings.append(np.ceil(k/2))
        phidden['paddings'] = paddings
        super(ConvolutionalLatent, self).__init__(pins, pouts, phidden=phidden, *args, **kwargs)
        mlp_input_size = int(reduce(lambda x, y: x*y, self.get_output_conv_length()))
        
#        # make mlp
#        if issubclass(type(pouts),  list):
#            n_outs = sum([x['dim'] for x in pouts])
        self.post_encoder = self.get_post_mlp(mlp_input_size*self.n_channels[-1], self.pouts, *args, **kwargs)
    
    def get_post_mlp(self, input_dim, pouts, nn_lin="ReLU", *args, **kwargs):   
        return MLP(input_dim, pouts, nn_lin=nn_lin, phidden={'nlayers':1, 'dim':800}, *args, **kwargs)
    
    def forward(self, x):
        mlp_input_size = reduce(lambda x, y: x*y, self.get_output_conv_length())
        conv_out = super(ConvolutionalLatent, self).forward(x)
        conv_out = conv_out.reshape(x.shape[0], mlp_input_size*self.n_channels[-1])
        out = self.post_encoder(conv_out)
        return out
    
    
class DeconvolutionalLatent(Convolutional):
    conv_module = DeconvLayer
    def __init__(self, pouts, pins, phidden={"dim":[64,32,8], "conv_dim":2, "kernel_size":[16,8,4]}, encoder=None, nn_lin="ReLU", *args, **kwargs):
        assert not issubclass(type(pins), list) or len(pins)==1
        if issubclass(type(pins), list):
            pins = dict(pins[0])

        phidden = dict(phidden)
        phidden['dim'] = list(reversed(phidden['dim'])) + [1]
        phidden['kernel_size'] = list(reversed(phidden['kernel_size']))
        phidden['pool'] = list(reversed(phidden['pool']))


        if not issubclass(type(nn_lin), list):
            nn_lin = [nn_lin]*len(phidden['kernel_size'])
        if pins['dist'] == dist.Bernoulli:
            nn_lin[-1] = 'Sigmoid'
        else:
            nn_lin[-1] = None
        
        super(DeconvolutionalLatent, self).__init__(pouts, pins, nn_lin=nn_lin, phidden = phidden, *args, **kwargs)
        #self.encoder = encoder

        self.output_size = encoder.get_output_conv_length()
        if encoder is None or not encoder.return_indices:
            raise Warning('Deconvolutional module has to be initialized with a valid Convolutional module in order to perform appropriate Unpooling.')
        mlp_output_size = int(reduce(lambda x, y: x*y, self.output_size)*phidden['dim'][0])
        
        self.pre_mlp = self.get_pre_mlp(pouts, mlp_output_size, *args, **kwargs)
#        self.pre_encoder = 
        
    def get_pre_mlp(self, pin, output_dim, nn_lin="ReLU", *args, **kwargs):   
        return MLP(pin, output_dim, nn_lin=nn_lin, phidden={'nlayers':1, 'dim':800}, *args, **kwargs)
        
    def forward(self, x):
        pre_output = self.pre_mlp(x)
        original_size = self.output_size
        pre_output = pre_output.reshape(x.shape[0], self.n_channels[0], *original_size)
        # successively pass into convolutional modules
        #indices = self.encoder.get_pooling_indices()
        out = pre_output
        for l in range(self.depth):
            #out = self.conv_encoders[l](out, indices[-(l+1)])
            out = self.conv_encoders[l](out)

        return [[torch.squeeze(out)]]
