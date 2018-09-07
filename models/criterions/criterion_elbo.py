import torch, pdb
import numpy as np
from . import log_density, kld
from ..distributions.priors import IsotropicGaussian
from .criterion_criterion import Criterion


def checklist(item):
    if not issubclass(type(item), list):
        item = [item]
    return item


class ELBO(Criterion):
    def __init__(self, options={}):
        super(ELBO, self).__init__(options)
        self.warmup = options.get('warmup', 100)
        self.beta = options.get('beta', 1.0)
        self.size_average = options.get('size_average', False)
        self.epoch = -1
        
    def get_reconstruction_error(self, model, x, out, *args, **kwargs):
        # reconstruction error
        if not issubclass(type(x), list):
            x = [x]
        rec_error = torch.zeros(1, requires_grad=True, device=model.device)
        for i in range(len(x)):
            rec_error = rec_error + log_density(model.pinput[i]['dist'])(x[i], out['x_params'][i], size_average=self.size_average)
        return rec_error
    
    
    def get_kld(self, model, out, *args, **kwargs):
        kld_error = torch.zeros(1, requires_grad=True, device=model.device)
        for l, platent_tmp in enumerate(model.platent):
            # iterate over layers
            current_zenc_params = checklist(out['z_params_enc'][l])
            platent_tmp = checklist(platent_tmp)
            for i, pl in enumerate(platent_tmp):
                if l == len(model.platent)-1:
                    prior = pl.get('prior') or IsotropicGaussian(pl['dim'])
                    kld_error = kld_error + kld(pl["dist"], prior.dist)(current_zenc_params[i], prior.get_params(device=model.device, *args, **kwargs), size_average=self.size_average)
                else:
                    current_zdec_params = checklist(out['z_params_dec'][l])
                    kld_error = kld_error + kld(pl["dist"], pl["dist"])(current_zenc_params[i], current_zdec_params[i], size_average = self.size_average)
        return kld_error

    def get_montecarlo(self, model, out, *args, **kwargs):
        kld_error = torch.zeros(1, requires_grad=True, device=model.device)
        for l, platent_tmp in enumerate(model.platent):
            # iterate over layers
            # turn parameters into lists
            current_zenc =  checklist(out['z_enc'][l])                  # sampled z
            current_zenc_params = checklist(out['z_params_enc'][l])     # z params
            platent_tmp = checklist(platent_tmp)                        # current layer's parameters
            # enumerate over layers
            for i, pl in enumerate(platent_tmp):
                # enumerate over splitted latent variables 
                # get p(z | prior)
                if l == len(model.platent)-1:
                    prior = pl.get('prior') or IsotropicGaussian(pl['dim'])
                    log_p = log_density(prior.dist)(current_zenc[i], prior.get_params(device=model.device, *args, **kwargs))
                else:
                    current_zdec_params = checklist(out['z_params_dec'][l])
                    log_p = log_density(pl['dist'])(current_zenc[i], current_zdec_params[i], size_average=self.size_average)
                # get q(z | z_params)
                log_q = log_density(pl['dist'])(current_zenc[i],  current_zenc_params[l], size_average=self.size_average)
                # compute kld component
                kld_error = kld_error + log_p - log_q
                
        return kld_error

    
    def get_regularization_error(self, model, out, *args, **kwargs):
        sample = kwargs.get('sample', False)
        if sample:
            kld_error = self.get_montecarlo(model, out, *args, **kwargs)
        else:
            kld_error = self.get_kld(model, out, *args, **kwargs)
        return kld_error
    
            
    def loss(self, model, out, epoch = None, x = None, options = {}, write=None, *args, **kwargs):
        assert not x is None        
        beta = options.get('beta', 1.0)
        if not epoch is None and self.warmup != 0:
            beta = beta * min(epoch / self.warmup, 1.0)
        rec_error = self.get_reconstruction_error(model, x, out, *args, **kwargs)
        kld_error = self.get_regularization_error(model, out, *args, **kwargs)
        loss = rec_error + beta * kld_error     
        losses = (rec_error, kld_error)
        return loss, losses
        
            
    def get_named_losses(self, losses):
        dict_losses = {'rec_loss':losses[0].item(), 'kld_loss':losses[1].item()}
        return dict_losses
    
    def get_min_loss(self, loss_names=None):
        if loss_names is None:
            loss_names = self.loss_history.keys()
        added_losses = []
        for loss_name in loss_names:
            added_losses.append(self.loss_history[loss_name])
        losses = np.sum(np.array(added_losses), 1)
        return torch.min(losses)
            
    
        
        
