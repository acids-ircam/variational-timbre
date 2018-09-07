#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:36:44 2017

@author: chemla
"""

from torch import Tensor
import torch.nn as nn
import pdb

from ..modules.modules_bottleneck import MLP, DLGMLayer
from .vae_vanillaVAE import VanillaVAE 

class VanillaDLGM(VanillaVAE):
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "layers":2}, *args, **kwargs):
        super(VanillaDLGM, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)
                
    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        encoders = nn.ModuleList()
        for i in range(len(latent_params)):
            if i == 0:
                encoders.append(self.make_encoder(input_params, latent_params[i], hidden_params[i], name="dlgm_encoder_%d"%i, *args, **kwargs))
            else:
                encoders.append(self.make_encoder(hidden_params[i-1], latent_params[i], hidden_params[i], name="dlgm_encoder_%d"%i, *args, **kwargs))
        return encoders
            
    def make_decoders(self, input_params, latent_params, hidden_params, top_linear=True, *args, **kwargs):   
        decoders = nn.ModuleList()
        top_dim = latent_params[-1]['dim'] if not issubclass(type(latent_params[-1]), list)  else sum([x['dim'] for x in latent_params[-1]])
        decoders.append(nn.Linear(top_dim, top_dim))
        for i in reversed(range(0, len(latent_params))):
            if i == 0:
                phidden_dec = dict(hidden_params[0]); phidden_dec['batch_norm']=False
                ModuleClass = phidden_dec.get('module', MLP)
                decoders.append(ModuleClass(latent_params[0], input_params, phidden_dec, name="dlgm_decoder_%d"%i))
            else:
                decoders.append(DLGMLayer(latent_params[i], latent_params[i-1], hidden_params[i], name="dlgm_decoder_%d"%i))
        return decoders


    # Process routines
    def encode(self, x, y=None, options={"sample":True}, encoders=None, platent=None, *args, **kwargs):
        if encoders is None:
            encoders = self.encoders
        if platent is None:
            platent = self.platent
        sample = options.get('sample') or True
        previous_output = x
        z_params = []; z = [];
        for i in range(0, len(platent)):
            params, h = encoders[i](previous_output, outputHidden=True)
            z_params.append(params); 
            if sample:
                if not issubclass(type(z_params[i]), list):
                    z.append(platent[i]['dist'](*params).rsample())      
                else:
                    z_tmp = []
#                    print(params)
                    for j, p_tmp in enumerate(params):
                        z_tmp.append(platent[i][j]['dist'](*params[j]).rsample())
                    z.append(z_tmp)
            else:
                z.append(z_params[i][0])
            previous_output = h
        return z_params, z
    
    
    def decode(self, z_enc=[], y=None, options={"sample":True}, decoders=None, *args, **kwargs):
        sample = options.get('sample') or True
        if decoders is None:
            decoders = self.decoders
        if not issubclass(type(z_enc), list):
            z_enc = [z_enc]
        z_params = []; z_dec = []; 
        if z_enc == []:
            prior_params = self.platent[-1]["prior"]["params"]
            if self.is_cuda:
                prior_params = [x.cuda() for x in prior_params]
            z_enc.append(self.platent[-1]["dist"](*prior_params))

#        ipdb.set_trace()
        z_dec.append(decoders[0](z_enc[-1]))
        for layer in range(1, len(self.platent)):
            try:
                current_z_enc = z_enc[-layer-1]
            except:
                current_z_enc = Tensor(z_enc[-1].size(0), self.platent[-layer-1]['dim'], requires_grad=True).zero_()
                if self.is_cuda:
                    current_z_enc = current_z_enc.cuda()
            z, params = decoders[layer](z_dec[-1], current_z_enc)
            z_dec.append(z)
            z_params.append(params)
            
        z_dec = [z for z in reversed(z_dec)]
        z_params = [z for z in reversed(z_params)]
        x_params = self.decoders[-1].forward(z_dec[0])
        return x_params, z_params, z_dec
    

