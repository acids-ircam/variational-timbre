#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:19:14 2017

@author: chemla
"""
#import torch
#from torch.autograd import Variable

import torch
from torch.autograd import Variable
import numpy as np
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition
from mpl_toolkits.mplot3d import Axes3D
from utils.onehot import fromOneHot
import matplotlib.patches as mpatches

try:
    from matplotlib import pyplot as plt
except:
    import matplotlib 
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    
    
from numpy.random import permutation
from utils.onehot import oneHot


#######################################################################
##########      Transformations
###

class Embedding(object):
    def __init__(self, *args, **kwargs):
        pass 
    
    def transform(*args, **kwargs):
        return np.array([])
        
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
        
class PCA(decomposition.PCA, Embedding):
    invertible = True
    def __init__(self, *args, **kwargs):
        super(PCA, self).__init__(*args, **kwargs)

class LocallyLinearEmbedding(manifold.LocallyLinearEmbedding, Embedding):
    invertible = False
    def __init__(self, *args, **kwargs):
        super(LocallyLinearEmbedding, self).__init__(*args, **kwargs)

class MDS(manifold.MDS, Embedding):
    invertible = False
    def __init__(self, *args, **kwargs):
        super(MDS, self).__init__(*args, **kwargs)

class TSNE(manifold.TSNE, Embedding):
    invertible = False
    def __init__(self, *args, **kwargs):
        super(TSNE, self).__init__(*args, **kwargs)




class DimensionReduction(Embedding):
    invertible = True
    def __init__(self, *args, fill_value = 0, **kwargs):
        super(DimensionReduction, self).__init__(*args, **kwargs)
        self.dimensions = np.array(args)
        self.fill_value = 0
        self.init_dim = None
        
    def fit(self, z):
        if z.ndim == 1:
            self.init_dim = z.shape[0]
        else:
            self.init_dim = z.shape[-1]
    
    def transform(self, z):
        return z[self.dimensions]
    
    def inverse_transform(self, z, target_dim=None):
        if target_dim is None:
            target_dim = self.init_dim
        if self.init_dim is None:
            raise Exception('[Warning] Please inform a target dimension to invert data.')
        if z.ndim == 1:
            invert_z = np.zeros(target_dim)
            invert_z[self.dimensions] = z
        else:
            invert_z = np.zeros((z.shape[0], target_dim))
            invert_z[:, self.dimensions] = np.squeeze(z)
        return invert_z
    
            
        
    
    
#######################################################################
##########      Manifold (points + transformation)
###
        
class Manifold(object):
    def __init__(self, transformation, z):
        super(Manifold, self).__init__()
        self.transformation = transformation
        self.orig_z = z
        self.z = transformation.transform(z)
        self.ids = None
        



    


class LatentManifold(Manifold):
    def __init__(self, transformation, model, dataset, sample=False, metadata=None, layer=0, ids=None, *args, **kwargs):
        self.model = model
        self.transformation = transformation
        self.ids = dataset.shape[0] if ids is None else ids
        data = dataset[self.ids]
        metadata = metadata[self.ids]

        data = model.format_input_data(data)
        metadata = model.format_label_data(metadata)
        out, _ = model.encode(data, y=metadata)
        self.params = out[layer]
        if sample:
            # sample from distributions
            self.orig_z = model.platent[layer]['dist'](*out[layer]).sample()
        else:
            # sample mean
            self.orig_z = model.platent[layer]['dist'](*out[layer]).sample()
        self.orig_z = self.orig_z.cpu().detach().numpy()
        self.z = transformation.fit_transform(self.orig_z)
        
        
