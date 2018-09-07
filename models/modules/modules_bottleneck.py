#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:49:58 2017

@author: chemla
"""
import pdb
from torch import cat
import torch.nn as nn
from collections import OrderedDict

import sys
sys.path.append('../..')

from .utils import init_module
from .modules_distribution import get_module_from_density

# Full modules for variational algorithms


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nn_lin="ReLU", batch_norm=True, dropout=None, name_suffix="", *args, **kwargs):
        super(MLPLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        modules = OrderedDict()
        modules["hidden"+name_suffix] =  nn.Linear(input_dim, output_dim)
        init_module(modules["hidden"+name_suffix], nn_lin)
        if batch_norm:
            modules["batch_norm_"+name_suffix]= nn.BatchNorm1d(output_dim)
        if not dropout is None:
            modules['dropout_'+name_suffix] = nn.Dropout(dropout)
        modules["nnlin"+name_suffix] = getattr(nn, nn_lin)()
        self.module = nn.Sequential(modules)
        
    def forward(self, x):
        return self.module(x)


class MLPResidualLayer(MLPLayer):
    def forward(self, x):
        out = self.module(x)
        if self.input_dim == self.output_dim:
            out = nn.functional.relu(out + x)
        return out


class MLP(nn.Module):
    ''' Generic layer that is used by generative variational models as encoders, decoders or only hidden layers.'''
    def __init__(self, pins, pouts=None, phidden={"dim":800, "nlayers":2}, nn_lin="ReLU", name="", *args, **kwargs):
        ''':param pins: Input properties.
        :type pins: dict or [dict]
        :param pouts: Out propoerties. Leave to None if you only want hidden modules.
        :type pouts: [dict] or None
        :param phidden: properties of hidden layers.
        :type phidden: dict
        :param nn_lin: name of non-linear layer 
        :type nn_lin: string
        :param name: name of module
        :type name: string'''
        # Configurations
        super(MLP, self).__init__()
        self.phidden = phidden
        if not issubclass(type(pins), list):
            pins = [pins]
        self.input_params = pins
        
        # get hidden layers
        self.hidden_module = self.get_hidden_layers(pins, phidden, nn_lin, name)
        
        # get output layers
        if pouts!=None:
            self.out_modules = self.get_output_layers(phidden, pouts)
        else:
            self.out_modules=None
        self.latent_params = pouts
            
            
    @classmethod
    def get_hidden_layers(cls, pins, phidden={"dim":800, "nlayers":2, "batch_norm":False}, nn_lin="ReLU", name=""):
        '''outputs the hidden module of the layer.
        :param input_dim: dimension of the input
        :type input_dim: int
        :param phidden: parameters of hidden layers
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str
        :returns: nn.Sequential'''
        # Hidden layers
        input_dim = 0
        for p in pins:
            new_input = p['dim'] if issubclass(type(p), dict) else p
            input_dim += new_input
            
        residual = phidden.get("residual", False)
        LayerModule = MLPResidualLayer if residual else MLPLayer
        
        if not issubclass(type(phidden), list):       
            modules = OrderedDict()
            nlayers = phidden.get('nlayers', 1)
            hidden_dim = phidden.get('dim',800)
            for i in range(nlayers):
                n_in = int(input_dim) if i==0 else int(hidden_dim)
                modules['layer_%d'%i] = LayerModule(n_in, int(hidden_dim), nn_lin=nn_lin, batch_norm=phidden.get('batch_norm', False), dropout = phidden.get('dropout'), name_suffix="_%d"%i)
            return nn.Sequential(modules)

        else:
            modules = nn.ModuleList()
            for i, ph in enumerate(phidden):
                mod = OrderedDict()
                nlayers = ph.get('nlayers', 1)
                hidden_dim = phidden.get('dim',800)
                for i in range(ph['nlayers']):
                    n_in = int(input_dim) if i==0 else int(hidden_dim)
                    mod['layer_%d'%i] = LayerModule(n_in, int(hidden_dim), nn_lin=nn_lin, batch_norm=ph.get('batch_norm', False), dropout = ph.get('dropout', None), name_suffix="_%d"%i)
                modules.append(nn.Sequential(mod))
            return modules
                
    def get_output_layers(self, pin, pouts):
        '''returns output layers with resepct to the output distribution
        :param in_dim: dimension of input
        :type in_dim: int
        :param pouts: properties of outputs
        :type pouts: dict or [dict]
        :returns: ModuleList'''
        out_modules=[]
        if not issubclass(type(pouts), list):
            pouts = [pouts]
        for pout in pouts:
            if issubclass(type(pout),  dict):
                out_modules.append(get_module_from_density(pout["dist"])(pin, pout))
            else:
                out_modules.append(nn.Linear(pin['dim'], pout))
        return nn.ModuleList(out_modules)
    
    
    def forward(self, x, outputHidden=False):
        '''outputs parameters of corresponding output distributions
        :param x: input or vector of inputs.
        :type x: torch.Tensor or [torch.Tensor ... torch.Tensor]
        :param outputHidden: also outputs hidden vector
        :type outputHidden: True
        :returns: (torch.Tensor..torch.Tensor)[, torch.Tensor]'''
        if type(x)==list:
            ins = cat(x, 1)
        else:
            ins = x
            
        if issubclass(type(self.hidden_module), nn.ModuleList):
            h = []
            for i, x_tmp in enumerate(x):
                h.append(self.hidden_module[i](ins))
        else:
            h = self.hidden_module(ins)
            
        z = []
        if self.out_modules!=None:
            for i in self.out_modules:
                if issubclass(type(self.hidden_module), nn.ModuleList):
                    for j, h_tmp in h:
                        z.append(i(h_tmp))
                else: 
                    z.append(i(h)) 
            if not issubclass(type(self.latent_params), list):
                z = z[0]
        if outputHidden:
            if self.out_modules!=None:
                return z,h
            else:
                return h
        else:
            if self.out_modules!=None:
                return z
            else:
                return h    


#class ResidualMLP(MLP):
#    @classmethod
#    def make_layer(self, input_dim, output_dim, nn_lin="ReLU", batch_norm=True, name_suffix=""):
#        modules = OrderedDict()
#        modules["hidden"+name_suffix] =  nn.Linear(input_dim, output_dim)
#        init_module(modules["hidden"+name_suffix], nn_lin)
#        if batch_norm:
#            modules["batch_norm_"+name_suffix]= nn.BatchNorm1d(output_dim)
#        modules["nnlin"+name_suffix] = getattr(nn, nn_lin)()
#        if input_dim == output_dim:
#            # get some residual
#            
#        return nn.Sequential(modules)
#    
    
    
    
#class Convolutional(nn.Module)    
#    def __init__(self, pins, pouts, phidden={}, nn_lin="ReLU", name=""):
        
    
    
class DLGMLayer(nn.Module):
    ''' Specific decoding module for Deep Latent Gaussian Models'''
    def __init__(self, pins, pouts, phidden={"dim":800, "nlayers":2}, nn_lin="ReLU", name=""):
        '''
        :param pins: parameters of the above layer
        :type pins: dict
        :param pouts: parameters of the ouput distribution
        :type pouts: dict
        :param phidden: parameters of the hidden layer(s)
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str'''
        if not issubclass(type(pins), list):
            pins = [pins]
        super(DLGMLayer, self).__init__()
        self.hidden_module = MLP.get_hidden_layers(pins, phidden=phidden, nn_lin=nn_lin, name=name)
        if issubclass(type(pouts), list):
            self.out_module = nn.ModuleList()
            self.cov_module = nn.ModuleList()
            for pout in pouts:
                self.out_module.append(nn.Linear(phidden['dim'], pout['dim']))
                init_module(self.out_module, 'Linear')
                self.cov_module.append(nn.Sequential(nn.Linear(pout['dim'], pout['dim']), nn.Sigmoid()))
                init_module(self.cov_module, 'Sigmoid')
        else:
            self.out_module = nn.Linear(phidden['dim'], pouts['dim'])
            init_module(self.out_module, 'Linear')
            self.cov_module = nn.Sequential(nn.Linear(pouts['dim'], pouts['dim']), nn.Sigmoid())
            init_module(self.cov_module, 'Sigmoid')
        
    def forward(self, z, eps):
        '''outputs the latent vector of the corresponding layer
        :param z: latent vector of the above layer
        :type z: torch.Tensor
        :param eps: latent stochastic variables
        :type z: torch.Tensor
        :returns:torch.Tensor'''
        if issubclass(type(z), list):
            z = cat(tuple(z), 1)
        out_h = self.hidden_module(z)
        if issubclass(type(self.out_module), nn.ModuleList):
            out = []; mean = []; std = []
            for i, module in enumerate(self.out_module):
                mean.append(self.out_module[i](out_h))
                std.append(self.cov_module[i](eps))
                out.append(mean + std)
        else:
            mean = self.out_module(out_h)
            std = self.cov_module(eps)
            out = mean + std
        return out, (mean, std)
        
    
class DiscriminatorLayer(nn.Module):
    def __init__(self, pins, phidden={"dim":800, "nlayers":2},  name="", nn_lin="ReLU"):
        super(DiscriminatorLayer, self).__init__()
        if not issubclass(type(pins), list):
            pins = [pins]
        self.hidden_module = nn.Sequential(MLP(pins, phidden=phidden, pouts=None, nn_lin=nn_lin),
                                           nn.Linear(phidden['dim'], 1), nn.Sigmoid())
        init_module(self.hidden_module,'Sigmoid')
        
    def forward(self, z):
        return self.hidden_module(z)
        
        
