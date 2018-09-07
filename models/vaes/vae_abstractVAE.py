#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:26:20 2018

@author: chemla
"""
import numpy as np
import torch
import torch.nn as nn
import pdb
from collections import OrderedDict
from utils import oneHot
#import visualize.dimension_reduction as dr

class AbstractVAE(nn.Module):
    
    #############################################
    ###  Architecture methods
    
    def __len__(self):
        return len(self.latent_params)
    
    def __init__(self, input_params, latent_params, hidden_params=[{"dim":800, "layers":2}], optim_params = {}, *args, **kwargs):
        super(AbstractVAE, self).__init__()
        
        # global attributes
        self.is_cuda = False
        self.device='cpu'
        self.dump_patches = True

        # retaining constructor for save & load routines (stupid with dump_patches?)
        if not hasattr(self, 'constructor'):
            self.constructor = {'input_params':input_params, 'latent_params':latent_params, 'hidden_params':hidden_params,
                                'optim_params':optim_params, 'args':args, 'kwargs':kwargs} # remind construction arguments for easy load/save
        
        # turn singleton specifications into lists
        if not issubclass(type(input_params), list):
            input_params = [input_params]
        if not issubclass(type(latent_params), list):
            latent_params = [latent_params]
        if not issubclass(type(hidden_params), list):
            hidden_params = [hidden_params]
            
        # check that hidden layers' specifications are well-numbered
        if len(hidden_params) < len(latent_params):
            print("[Warning] hidden layers specifcations is under-complete. Copying last configurations for missing layers")
            last_layer = hidden_params[-1]
            while len(hidden_params) < len(latent_params):
                hidden_params.append(last_layer)
                
        self.pinput = input_params; self.phidden = hidden_params; self.platent = latent_params
        self.init_modules(self.pinput, self.platent, self.phidden, *args, **kwargs)
        
        # init optimizer
        self.init_optimizer(optim_params)
        self.optim_params = optim_params 
        
    # architecture methods
    def init_modules(self, input_params, latent_params, hidden_params, *args, **kwargs):
        hidden_dec_params = []
        hidden_enc_params = []
        for l in range(len(hidden_params)):
            hidden_dec_params.append(hidden_params[l].get('decoders', False) or hidden_params[l]) 
            hidden_enc_params.append(hidden_params[l].get('encoders', False) or hidden_params[l])
        self.encoders = self.make_encoders(input_params, latent_params, hidden_enc_params, *args, **kwargs)
        self.decoders = self.make_decoders(input_params, latent_params, hidden_dec_params, *args, **kwargs)
        
    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        return nn.ModuleList()
    
    def make_decoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        return nn.ModuleList()
    

    #############################################
    ###  Processing methods
    
    def encode(self, x, options={}, *args, **kwargs):
        return None
    
    def decode(self, z, options={}, *args, **kwargs):
        return None
    
    def sample(self, z, options={}, *args, **kwargs):
        return None
    
    def forward(self, x, y=None, options={}, *args, **kwargs):
        return {}
    
    def make_dict_from_losses(self, losses):
        dict_losses = {}
        for i in range(len(losses)):
            loss = losses[i]
            dict_losses['losses_%d'%i] = loss.item()
        return dict_losses
    
    # this is a global step function, where the optimize function should be overloaded
    def step(self, loss, options={'epoch':0}, *args, **kwargs):
        # update optimizers in case
        self.update_optimizers(options)        
        # optimize 
        self.optimize(loss)
        pass
        

    
    #############################################
    ###  Loss methods & optimization schemes
    
    def get_loss(self, *args):
        return 0.

    def init_optimizer(self, optim_params={}):
#        self.optimizers = {'default':getattr(torch.optim, optimMethod)(self.parameters(), **optimArgs)}
        self.optimizers = {} # optimizers is here a dictionary, in case of multi-step optimization
        self.schedulers = {}       
            
    def update_optimizers(self, options):
        # optimizer update at each 'step' call
        for _, o in self.optimizers.items():
            o.zero_grad() # zero gradients    
            
    def optimize(self, loss, *args, **kwargs):
        pass

    def schedule(self, *args, **kwargs):
        pass

    #############################################
    ###  Loss methods & optimization schemes

    def cuda(self, device=None):
        if device is None: 
            device = torch.cuda.current_device()
        self.device = torch.device('cuda:%d'%device)
        self.is_cuda = True
        super(AbstractVAE, self).cuda(device)

    def cpu(self):
        self.device = -1
        self.is_cuda = False
        super(AbstractVAE, self).cpu()




    #############################################
    ###  Load / save methods
    
    def save(self, filename, *args, **kwargs):
        if self.is_cuda:
            state_dict = OrderedDict(self.state_dict())
            for i, k in state_dict.items():
                state_dict[i] = k.cpu()
        else:
            state_dict = self.state_dict()
        constructor = dict(self.constructor)
        save = {'state_dict':state_dict, 'init_args':constructor, 'class':self.__class__, 
                'optimizers':self.optimizers, 'schedulers':self.schedulers}
        for k,v in kwargs.items():
            save[k] = v
        torch.save(save, filename)
        
    @classmethod
    def load(cls, pickle, with_optimizer=False):
        init_args = pickle['init_args']
        for k,v in init_args['kwargs'].items():
            init_args[k] = v
        del init_args['kwargs']
        vae = cls(**pickle['init_args'])
        vae.load_state_dict(pickle['state_dict'])
        if with_optimizer:
            vae.optimizers = pickle.get('optimizers', {})
            vae.schedulers = pickle.get('optimizers', {})
        else:
            vae.init_optimizer(vae.optim_params)

        return vae



    #############################################
    ###  Utility methods
    
    def format_input_data(self, x, requires_grad=True, *args, **kwargs):
        if x is None:
            return 
        if issubclass(type(x), list):
            x = [x for x in map(self.format_input_data, x)]
        else:
            if type(x)==list:
                x = np.array(x)
            if type(x)==np.ndarray:
                if x.dtype!='float32':
                    x = x.astype('float32')
                x = torch.from_numpy(x)
            x = x.to(self.device, dtype=torch.float32)
        for i in x:
            i.requires_grad_(requires_grad)
        return x
    
    def format_label_data(self, x, requires_grad=False, plabel=None, *args, **kwargs):
        if x is None:
            return 
        if plabel is None:
            plabel = self.plabel
        if issubclass(type(x), list):
            new_x = []
            for i, label in enumerate(x):
                new_x.append(self.format_label_data(label, plabel=plabel[i]))
            x = new_x
        else:
            if type(x)==list:
                x = np.array(x)
            if type(x)==np.ndarray:
                if x.ndim == 1:
                    if issubclass(type(plabel), list):
                        label_dim = 0
                        for pl in plabel:
                            label_dim += pl['dim']
                    else:
                        label_dim = plabel['dim']
                    x = oneHot(x, label_dim)
                if x.dtype!='float32':
                    x = x.astype('float32')
                x = torch.from_numpy(x)
            x = x.to(self.device, dtype=torch.float32)
        for i in x:
            i.requires_grad_(requires_grad)
        return x

       

    #############################################
    ###  Manifold methods
    
#    def make_manifold(self, name, dataset, method=dr.latent_pca, task=None, n_points=None, layer=-1, options={'n_components':2}, *args, **kwargs):
#        if n_points is None:
#            n_points = len(dataset.data)
#        ids  = np.random.permutation(len(dataset.data))[:n_points]
#        data = dataset.data[ids, :]
#        if not task is None:
#            y = dataset.metadata[task][ids]
#        else:
#            y = None
#        
#        _, z = self.encode(data, y=y, *args, **kwargs)
#        _, self.manifolds[name] = method(z[layer].data.numpy(), **options)
#    
