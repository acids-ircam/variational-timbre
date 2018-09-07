#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:26:17 2018

@author: chemla
"""

import copy, pdb
import numpy as np

class Criterion(object):
    def __init__(self, options={}, weight=1.0):
        super(Criterion, self).__init__()
        self.weight = weight
        self.loss_history = {}
        
    def loss(self, *args, options={}, **kwargs):
        return 0.

    def write(self, name, losses):
        losses = [l.detach().cpu().numpy() for l in losses]
        if not name in self.loss_history.keys():
            self.loss_history[name] = []
        self.loss_history[name].append(losses)

    def get_named_losses(self, losses):
        return {}
    
    def __call__(self, *args, options={}, **kwargs):
        l, losses = self.loss(*args, options={}, **kwargs)
        return self.weight * l, losses
        
    def __add__(self, c):
        if issubclass(type(c), LossContainer):
            nc = copy.deepcopy(c)
            nc.criterions_.append(self)
        if issubclass(type(c), Criterion):
            c = copy.deepcopy(c)
            nc = LossContainer([self, c])
        return nc
    
    def __radd__(self, c):
        return self.__add__(c)
        
    def __sub__(self, c):
        if issubclass(type(c), LossContainer):
            nc = copy.deepcopy(c)
            c.weight = -1.0
            nc.criterions_.append(self)
        if issubclass(type(c), Criterion):
            c = copy.deepcopy(c)
            c.weight = -1.0
            nc = LossContainer([self, c])
        return nc
    
    def __rsub__(self, c):
        return self.__sub__(c)
        
    def __mul__(self, f):
        assert issubclass(type(f), float) or issubclass(type(f), np.ndarray)
        new = copy.deepcopy(self)
        new.weight *= f
        return new
        
    def __rmul__(self, c):
        return self.__mul__(c)
 
    def __div__(self, f):
        assert issubclass(type(f), float)
        new = copy.deepcopy(self)
        new.weight /= f
        return new
        
    def __rdiv__(self, c):
        return self.__div__(c)
        
     
class LossContainer(Criterion):
    def __init__(self, criterions=[], options={}, weight=1.0):
        super(Criterion, self).__init__()
        self.criterions_ = criterions
        self.loss_history = {}
    
    def loss(self, *args, options={}, **kwargs):
        full_losses = [c(*args, options=options, **kwargs) for c in self.criterions_]
        loss = 0.; losses = []
        for l, ls in full_losses:
            loss = loss + l
            losses.append(ls)
        return loss, losses
    
    def get_named_losses(self, losses):
        named_losses=dict()
        for i, l in enumerate(losses):
            current_loss = self.criterions_[i].get_named_losses(l)
            named_losses = {**named_losses, **current_loss}
        return named_losses

    def write(self, name, losses):
        for i, criterion in enumerate(self.criterions_):
            criterion.write(name, losses[i])

            
            

