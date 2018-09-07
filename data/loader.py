#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:10:03 2018

@author: chemla
"""

import os, os.path
from .audio import DatasetAudio


def merge_dicts(dict1, dict2, verbose=False):
    new_dict = dict(dict1)
    for k, v in dict2.items():
        if verbose:
            if k in dict1.keys():
                print('[Warning] key %s present in both dictionaries'%k)
        new_dict[k] = v
    return new_dict

def load_dataset(folder_path, analysis_path=None, transformType='stft', flattening_function=None, *args, **kwargs):
    folder_path = os.path.abspath(folder_path)
    _, dataset_name = os.path.split(folder_path)
    analysis_path = analysis_path or '/tmp/'+dataset_name
    flattening_function = flattening_function or (lambda x: x[0])
    importOptions = {
      'dataPrefix': folder_path,
      'dataDirectory':folder_path,
      'analysisDirectory':analysis_path, # Root to place (and find) the transformed data
      'transformName':transformType,
      'importType':[],                                        # Type of import (direct or asynchronous)
      'importCallback':None,                                  # Function to perform import of data
      'types':['mp3', 'wav', 'wave', 'aif', 'aiff', 'au'],    # Accepted types of files
      'transformCallback':None,                               # Function to transform data (can be a list)
      'verbose':True,                                         # Be verbose or not
      'checkIntegrity':True,                                  # Check that files exist (while loading)
      'forceUpdate':True,                                     # Force the update
      'matlabCommand':'/usr/local/MATLAB/MATLAB_Production_Server/R2015a/bin/matlab',
    };
    dataset = DatasetAudio(importOptions)
    dataset.listDirectory()
    dataset.importMetadataTasks()
    transformList, transformParameters = dataset.getTransforms();
    
    # Compute Transforms
    transformList, transformParameters = dataset.getTransforms();
    transformParameters = merge_dicts(transformParameters, kwargs)
    transformOptions = dict(importOptions)
    transformOptions['transformTypes'] = [transformType]
    transformOptions['transformNames'] = [transformType] 
    transformOptions['transformParameters'] = [transformParameters]
    dataset.computeTransforms(None, transformOptions, padding=False)
    
    # Import transforms
    dataset.importData(None, importOptions);
    dataset.flattenData(flattening_function)
    dataset.constructPartition([], ['train','test'], [0.8, 0.2], False);
    return dataset