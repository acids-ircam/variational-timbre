#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:37:27 2017

@author: chemla
"""
import pdb, numpy as np
import torch
from .vae_vanillaDLGM import VanillaDLGM 
from utils.onehot import oneHot, fromOneHot
from copy import deepcopy


class ConditionalVAE(VanillaDLGM):
    def __init__(self, input_params, latent_params, hidden_params=[{"dim":800, "layers":2}], label_params=None, *args, **kwargs):
        assert not label_params is None
        self.plabel = label_params
        if not issubclass(type(self.plabel), list):
            self.plabel = [self.plabel]
        super(ConditionalVAE, self).__init__(input_params, latent_params, hidden_params=hidden_params, label_params=self.plabel, *args, **kwargs)
    
    def make_encoders(self, input_params, latent_params, hidden_params, label_params, *args, **kwargs):
        # add label to input of the encoder
        if not issubclass(type(input_params), list):
            enc_input = [input_params]
        else:
            enc_input = list(input_params)
        enc_input += label_params
        return super(ConditionalVAE, self).make_encoders(enc_input, latent_params, hidden_params)
        
    def make_decoders(self, input_params, latent_params, hidden_params, label_params, *args, **kwargs):
        # add label to input of the decoder
        if not issubclass(type(latent_params), list):
            latent_params_dec = [latent_params]
        else:
            latent_params_dec = list(latent_params)
        if issubclass(type(latent_params_dec[-1]), list):
            latent_params_dec[-1] += label_params
        else:
            latent_params_dec[-1] = [latent_params_dec[-1]] + label_params
        return super(ConditionalVAE, self).make_decoders(input_params, latent_params_dec, hidden_params)
    
#    def forward(self, x, y, options={}, *args, **kwargs):
#        z_params_enc, z_enc = self.encode(x, y=y, *args, **kwargs)
#        x_params, z_params_dec, z_dec = self.decode(z_enc, y=y, *args, **kwargs)
#        return {'x_params':x_params, 'z_params_dec':z_params_dec, 'z_dec':z_dec,
#                'z_params_enc':z_params_enc, 'z_enc':z_enc}
        
    def encode(self, x, y=None, options={}, *args, **kwargs):
        if not issubclass(type(x), list):
            x = [x]
        if y is None:
            raise Exception('Conditional VAE must be given label information')
        if not issubclass(type(y), list):
            y = [y]
        x = tuple(x); y = tuple(y)
        inp = torch.cat((*x, *y), 1)
        outs = super(ConditionalVAE, self).encode(inp)
        return outs
    

    def decode(self, z, y=None, options={}, *args, **kwargs):
        if not issubclass(type(z), list):
            z = [z]
        if y is None:
            raise Exception('Conditional VAE must be given label information')
        if not issubclass(type(y), list):
            y = [y]
        y = tuple(y)
        z[-1] = torch.cat((*z, *y), 1)
        outs = super(ConditionalVAE, self).decode(z)
        return outs

    def forward(self, x, y=None, options={}, *args, **kwargs):
        if y is None:
            raise Exception('Conditional VAE must be given label information')
        x = self.format_input_data(x)
        y = self.format_label_data(y)        
        z_params_enc, z_enc = self.encode(x, y=y, *args, **kwargs)
        x_params, z_params_dec, z_dec = self.decode(z_enc, y=y, *args, **kwargs)
        return {'x_params':x_params, 
                'z_params_dec':z_params_dec, 'z_dec':z_dec,
                'z_params_enc':z_params_enc, 'z_enc':z_enc}
        
    