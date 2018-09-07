# -*- coding: utf-8 -*-

import pdb
from numpy.random import permutation
import numpy as np

def length(array):
    if issubclass(type(array), list):
        return len(array)
    elif issubclass(type(array), np.ndarray):
        return array.shape[0]

class DataLoader(object):
    def __init__(self, dataset, batch_size, task=None, partition=None, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        if partition is None:
            random_indices = permutation(length(dataset.data)) 
        else:
            partition_ids = dataset.partitions[partition]
            random_indices = partition_ids[permutation(len(partition_ids))]
        n_batches = len(random_indices)//batch_size
        self.random_ids = np.split(random_indices[:n_batches*batch_size], len(random_indices)//batch_size)
        self.task = task
    
    
    def __iter__(self):
        for i in range(len(self.random_ids)):
            self.current_ids = self.random_ids[i]
            if issubclass(type(self.dataset.data), list):
                x = [d[self.current_ids] for d in self.dataset.data]
            else:
                x = self.dataset.data[self.current_ids]
            if not self.task is None:
                y = self.dataset.metadata[self.task][self.current_ids]
            else:
                y = None
#                yield self.transform(self.dataset.data[self.random_ids[i]]), None
            yield x,y
        

class SemiSupervisedDataLoader(object):
    #TODO default supervised ids
    def __init__(self, dataset, batch_size, task=None, sup_ids=None, ratio = 0.2, partition=None, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        if partition is None:
            random_indices = permutation(len(dataset.data)) 
        else:
            partition_ids = dataset.partitions[partition]
            random_indices = partition_ids[permutation(len(partition_ids))]
            
        if sup_ids is None:
            n_sup = int(ratio*len(dataset.data))
            sup_ids = np.random.permutation(len(dataset.data))[:n_sup]
            
        filtered_ids = np.array([x for x in filter(lambda x: not x in sup_ids, random_indices)])
        self.random_ids = np.split(filtered_ids[:len(filtered_ids)//batch_size*batch_size], len(filtered_ids)//batch_size)
        self.sup_ids = np.split(sup_ids, len(sup_ids)//batch_size)
        self.task = task
            
    def __iter__(self):
        for i in range(len(self.sup_ids)):
            yield self.dataset.data[self.sup_ids[i]], self.dataset.metadata[self.task][self.sup_ids[i]]
        for i in range(len(self.random_ids)):
            yield self.dataset.data[self.random_ids[i]], None
        
