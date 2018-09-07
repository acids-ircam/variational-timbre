#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:16:27 2018

@author: chemla
"""
from numpy import max
from scipy.io import wavfile
import os

def openaudio(file):
    name, ext = os.path.splitext(file)
    import_name=file
    if ext != '.wav':
        os.system('ffmpeg -i %s /tmp/%s.wav'%(file, name))
        import_name = '/tmp/%s.wav'%name
    _, sig = wavfile.read(import_name)
    return sig[:,0]/max(sig)
        