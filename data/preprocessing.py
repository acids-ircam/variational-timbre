#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:26:51 2018

@author: chemla
"""

import numpy as np

class Magnitude(object):
    def __init__(self, dataset, preprocessing='none', min_threshold=1e-6, normalize=False):
        super(Magnitude, self).__init__()
        if preprocessing == 'none':
            pass
        elif preprocessing == 'log' or 'log1p':
            self.data_std = 1.0
        elif preprocessing == 'nlog or nlog1p':
            self.data_std = np.std(dataset.data)
        else:
            print('hello')
            raise Exception('preprocessing %s not recognized'%preprocessing)
        self.preprocessing = preprocessing
        self.min_threshold = min_threshold
        self.normalize = normalize
        self.meanData = None
        self.maxData = None
        
    def invert(self, data):
        if self.normalize:
            data *= self.maxData
            data += self.meanData
        if self.preprocessing == 'none':
            return data
        elif self.preprocessing == 'log':
            return np.exp(data)
        elif self.preprocessing == 'log1p':
            return np.exp(data) - 1
        elif self.preprocessing == 'nlog':
            return np.exp(data)*self.data_std
        elif self.preprocessing == 'nlog1p':
            return (np.exp(data)-1)*self.data_std
        else:
            raise Exception('something fucked up. normally it shouldnt.')
            
        
            
    def __call__(self, data, write=False):
        if issubclass(type(data), list):
            return [self(x) for x in data]
        if self.preprocessing == 'none':
            return np.abs(data)
        elif self.preprocessing == 'log':
            new_data = data.copy()
            new_data[new_data < self.min_threshold] = self.min_threshold
            new_data =  np.log(np.abs(data))
        elif self.preprocessing == 'log1p':
            new_data = data.copy()
            new_data[new_data < self.min_threshold] = self.min_threshold
            new_data =  np.log(1+np.abs(data))
        elif self.preprocessing == 'nlog':
            new_data = data.copy()
            new_data[new_data < self.min_threshold] = self.min_threshold
            new_data =  np.log(np.abs(data)/self.data_std)
        elif self.preprocessing == 'nlog1p':
            new_data = data.copy()
            new_data[new_data < self.min_threshold] = self.min_threshold
            new_data =  np.log(1+np.abs(data)/self.data_std)
        if self.normalize:
            if write or self.meanData is None:
                self.meanData = np.mean(new_data)
                new_data -= self.meanData
                self.maxData = np.max(np.abs(new_data))
                new_data /= self.maxData
            else:
                new_data -= self.meanData
                new_data /= self.maxData
        return new_data
            
        
    