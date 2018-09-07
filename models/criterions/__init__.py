
# -*-coding:utf-8 -*-
 
"""
    The ``criterions`` module
    ========================
 
    This package contains different criterions and criterion components for VAE training
 
    :Example:
 
    >>> from models.vaes import VanillaVAE
    >>> ELBO()
 
 
    Comments and issues
    ------------------------
        
        None for the moment
 
    Contributors
    ------------------------
        
    * Axel Chemla--Romeu-Santos (chemla@ircam.fr)
 
"""
 
# info
__version__ = "0.1.0"
__author__  = "chemla@ircam.fr"
__date__    = ""

# import sub modules
from .criterion_logdensities import *
from .criterion_klds import *
from .criterion_elbo import *