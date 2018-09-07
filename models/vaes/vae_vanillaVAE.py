#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:38:11 2017

@author: chemla
"""

import torch.nn as nn
import torch.optim
import pdb

from ..distributions.priors.prior_gaussians import IsotropicGaussian
from ..modules.modules_bottleneck import MLP
from . import AbstractVAE


class VanillaVAE(AbstractVAE):
    # initialisation of the VAE
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "nlayers":2}, *args, **kwargs):
        super(VanillaVAE, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)    
        
    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        encoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                encoders.append(self.make_encoder(input_params, latent_params[0], hidden_params[0], name="vae_encoder_%d"%layer))
            else:
                encoders.append(self.make_encoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], nn_lin="ReLU", name="vae_encoder_%d"%layer))
        return encoders
    
    @classmethod
    def make_encoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):               
        kwargs['name'] = kwargs.get('name', 'vae_encoder')
        ModuleClass = hidden_params.get('class', MLP)
#        module = latent_params.get('shared_encoder') or ModuleClass(input_params, latent_params, hidden_params, *args, **kwargs)
        module = ModuleClass(input_params, latent_params, hidden_params, *args, **kwargs)
        return module
    
    def make_decoders(self, input_params, latent_params, hidden_params, extra_inputs=[], *args, **kwargs):
        decoders = nn.ModuleList()
        for layer in reversed(range(len(latent_params))):
            if layer==0:
                #TODO pas terrible d'embarquer l'encoder comme ça
                new_decoder = self.make_decoder(input_params, latent_params[0], hidden_params[0], name="vae_decoder_%d"%layer, encoder = self.encoders[layer])
            else:
                new_decoder = self.make_decoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], name="vae_decoder_%d"%layer, encoder=self.encoders[layer])
            decoders.append(new_decoder)
        return decoders
    
    @classmethod
    def make_decoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'vae_decoder')
        ModuleClass = hidden_params.get('class', MLP)
        module = hidden_params.get('shared_decoder') or ModuleClass(latent_params, input_params, hidden_params, *args, **kwargs)
        module = ModuleClass(latent_params, input_params, hidden_params, *args, **kwargs)
        return module
        
    
    # processing methods
    def encode(self, x, sample=True, *args, **kwargs):
        z_params = []; z = []
        for layer in range(len(self.platent)+1):
            if layer == 0: 
                ins = x
            else:
                if not issubclass(type(self.platent[layer-1]), list):
                    if sample:
                        ins = self.platent[layer-1]["dist"](*z_params[layer-1]).rsample()
                    else:
                        ins = self.platent[layer-1]["dist"].analytic_mean(*z_params[layer-1], **kwargs).rsample()
                else:
                    ins = []
                    for i, latent_group in enumerate(self.platent[layer-1]):
                        if sample:
                            z_tmp = latent_group["dist"](*z_params[layer-1][i]).rsample()
                        else:
                            z_tmp = latent_group["dist"].analytic_mean(*z_params[layer-1][i]).rsample()
                        ins.append(z_tmp)
                z.append(ins)
            if layer < len(self.platent):
                z_params.append(self.encoders[layer].forward(ins))
        return z_params, z
    
    
    def decode(self, z_in=[], y=None, sample=True, path='encoder', layer=-1, *args, **kwargs):
        assert layer != 0
        # take latent position of upper layer in z_enc
        if not issubclass(type(z_in), list):
            z_in = [z_in]
        # from which stochastic layer do we sample first
        if layer<0:
            layer = len(self.platent) + layer + 1
        z_params_dec = []; z_dec = []; 
        for l in range(0, layer):
            # if the correspondant latent position is given, take it
            if l < len(z_in):
                current_z = z_in[-(l+1)]
            else:
                if len(z_params_dec) == 0:
                    # default case
                    current_z = self.platent[layer-l]['dist'](torch.zeros(z_in.shape))
                else:
                    # draw from previous decoder parameters
                    if sample: 
                        current_z = self.platent[layer-l]['dist'](*z_params_dec[-1])
                    else:
                        current_z = self.platent[layer-l]['dist'](*z_params_dec[-1])
            z_dec.append(current_z)
            params = self.decoders[len(self.platent)-layer+l](current_z)
            if l == layer-1:
                x_params = params
            else:
                z_params_dec.append(params)
        z_dec = [z for z in reversed(z_dec)]
        z_params_dec = [z for z in reversed(z_params_dec)]
        
        return x_params, z_params_dec, z_dec
       
    
    
    def forward(self, x, options={}, *args, **kwargs):
        x = self.format_input_data(x)
        z_params_enc, z_enc = self.encode(x, *args, **kwargs)
        x_params, z_params_dec, z_dec = self.decode(z_enc, *args, **kwargs)
        return {'x_params':x_params, 
                'z_params_dec':z_params_dec, 'z_dec':z_dec,
                'z_params_enc':z_params_enc, 'z_enc':z_enc}
    
    
    # define optimizer
    def init_optimizer(self, optim_params):
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-5})
        self.optimizers = {'default':getattr(torch.optim, alg)(self.parameters(), **optim_args)}   
        scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
        scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
        self.schedulers = {'default':getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['default'], **scheduler_args)} 
        
        
    def optimize(self, loss, options={}, *args, **kwargs):
        # optimize
        loss.backward()
        self.optimizers['default'].step()
        
    # define losses 
    

    
          
        

            
