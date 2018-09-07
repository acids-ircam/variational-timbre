#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:09:45 2018

@author: chemla
"""
import pdb
import torch, numpy as np
from .criterion_criterion import Criterion
from utils.onehot import fromOneHot




class PerceptiveL2Loss(Criterion):
    def __init__(self, centroids, targetDims, options={'normalize':False}):
        super(PerceptiveL2Loss, self).__init__()
        if issubclass(type(centroids), np.ndarray):
            self.centroids = torch.from_numpy(centroids).type('torch.FloatTensor')
        else:
            self.centroids = centroids.type('torch.FloatTensor')
        self.targetDims = targetDims
        self.normalize = options.get('normalize', False)
        
    def loss(self, model, out, y=None, layer=0, *args, **kwargs):
        assert not y is None
        z = out['z_enc'][layer]
        if y.dim() == 2:
            y = fromOneHot(y.cpu()); 
        # Create the target distance matrix
        targetDistMat = self.centroids[y, :][:, y]
        targetDistMat.requires_grad_(False)
        targetDistMat /= targetDistMat.max()
        if z.is_cuda:
            targetDistMat = targetDistMat.to(z.device)
        # Compute the dot product
        r = torch.mm(z[:, self.targetDims], z[:, self.targetDims].t())
        # Get the diagonal elements
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        # Compute the distance matrix in latent space
        actualDistMat = diag + diag.t() - 2*r + (1e-9)
        actualDistMat = actualDistMat.sqrt()
        if (self.normalize):
            actualDistMat = actualDistMat / (actualDistMat.max())
        # Compute difference to the desired distances
        res = torch.sqrt(torch.sum((actualDistMat - targetDistMat)**2))
        return res, (res, )
    
    def get_named_losses(self, losses):
        dict_losses = {'l2_perceptive':losses[0].item()}
        return dict_losses

    
    
    
class PerceptiveGaussianLoss(Criterion):
    def __init__(self, gaussianParams, targetDims, options={'normalize':False}):
        super(PerceptiveGaussianLoss, self).__init__()
        latent_means, latent_stds = gaussianParams
        self.latent_means = torch.from_numpy(latent_means).type('torch.FloatTensor') if issubclass(type(latent_means), np.ndarray) else latent_means.type('torch.FloatTensor')
        self.latent_stds = torch.from_numpy(latent_stds).type('torch.FloatTensor') if issubclass(type(latent_stds), np.ndarray) else latent_stds.type('torch.FloatTensor')
        self.targetDims = targetDims
        self.normalize = options.get('normalize', False)

    def loss(self, model, out, y=None, layer=0, *args, **kwargs):
        z = out['z_enc'][layer]
        if y.dim() == 2:
            y = fromOneHot(y); 
        # Create the target distance matrix
        targetMeans = self.latent_means[y, :][:, y] + 1e-5
        targetStds = self.latent_stds[y, :][:, y] + 1e-5
        targetMeans.requires_grad_(False); targetStds.requires_grad_(False)
        targetDistMat = torch.normal(targetMeans, targetStds)
        targetDistMat = targetDistMat / (targetDistMat.max())
        if z.is_cuda:
            targetDistMat = targetDistMat.to(z.device)
        # Compute the dot product
        r = torch.mm(z[:, self.targetDims], z[:, self.targetDims].t())
        # Get the diagonal elements
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        # Compute the distance matrix in latent space
        actualDistMat = diag + diag.t() - 2*r + (1e-7)
        actualDistMat = actualDistMat.sqrt()
        if (self.normalize):
            actualDistMat = actualDistMat / (actualDistMat.max())
        # Compute difference to the desired distances
        res = torch.sqrt(torch.sum((actualDistMat - targetDistMat)**2))
        return res, (res, )
    
    def get_named_losses(self, losses):
        dict_losses = {'gaussian_perceptive':losses[0].item()}
        return dict_losses




class PerceptiveStudent(Criterion):
    def __init__(self, centroids, targetDims, options={'normalize':False}):
        super(PerceptiveStudent, self).__init__()
        if issubclass(type(centroids), np.ndarray):
            self.centroids = torch.from_numpy(centroids).type('torch.FloatTensor')
        else:
            self.centroids = centroids.type('torch.FloatTensor')
        self.targetDims = targetDims
        self.normalize = options.get('normalize', False)
        
    def loss(self, model, out, y=None, layer=0, *args, **kwargs):
        z = out['z_enc'][layer]
        if y.dim() == 2:
            y = fromOneHot(y); 
        # Create the target distance matrix
        targetDistMat = self.centroids[y, :][:, y]
        targetDistMat.requires_grad_(False)
        targetDistMat = torch.pow((1 + targetDistMat), -1)
        targetDistMat = (targetDistMat / torch.sum(targetDistMat))
        #targetDistMat = targetDistMat / torch.max(targetDistMat)
        if z.is_cuda:
            targetDistMat = targetDistMat.to(z.device)
        # Compute the dot product
        r = torch.mm(z[:, self.targetDims], z[:, self.targetDims].t())
        # Get the diagonal elements
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        # Compute the distance matrix in latent space
        actualDistMat = diag + diag.t() - 2*r + (1e-9)
        # Final fronteer
        actualDistMat = torch.exp(- torch.sqrt(actualDistMat))
        actualDistMat = (actualDistMat / torch.sum(actualDistMat, 0))
        if (self.normalize):
            actualDistMat = actualDistMat / (actualDistMat.max())
        # Compute difference to the desired distances
        #res = torch.sum(((1 + ((actualDistMat - targetDistMat)**2)) ** -1)) #/ torch.sum(((1 + ((actualDistMat - targetDistMat)**2) ** -1))))
        res = torch.sum((actualDistMat) * torch.log(actualDistMat/targetDistMat))
        return res, (res, )
    
    def get_named_losses(self, losses):
        dict_losses = {'student_perceptive':losses[0].item()}
        return dict_losses
