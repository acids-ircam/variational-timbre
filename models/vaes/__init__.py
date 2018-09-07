
# -*-coding:utf-8 -*-
 
"""
    The ``vaes`` module
    ========================
 
    This package contains all vae models
 
    :Example:
 
    >>> from models.vaes import VanillaVAE
    >>> VanillaVAE()
 
    Subpackages available
    ---------------------

        * vaes
        * criterions
        * distributions
        * modules
 
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
from .vae_abstractVAE import AbstractVAE
from .vae_vanillaVAE import VanillaVAE
from .vae_vanillaDLGM import VanillaDLGM
from .vae_conditionalVAE import ConditionalVAE

