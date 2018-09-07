#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:27:58 2018

@author: chemla
"""

import argparse, os
import matplotlib
matplotlib.use('agg')

import torch, numpy as np
import torch.distributions as dist

import models.vaes as vaes
from models.criterions import ELBO

from utils.train import train_model

import misc.perceptive as percep
from models.criterions.criterion_perceptive import *
import models.distributions.priors.prior_gaussians as pg

CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))

#%% Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',           type=int,   default=-1,                                                     help='cuda ID device (leave -1 for cpu)')
parser.add_argument('--regularization', type=str,   default='l2', choices=['l2','prior','student', 'gaussian'],     help='perceptive regularization method')
parser.add_argument('--conditioning',   type=str,   default='pitch',                                                help='conditioning task')
parser.add_argument('--filter', type=int, default = 5)
parser.add_argument('--frames', type=int, nargs="*", default = [10, 25])

parser.add_argument('--target_dims',    type=str,   default='all',        choices=['min','max','all'],              help='regularization dimensions')
parser.add_argument('--mds_dims',       type=int,   default=3,                                                      help='perceptual MDS dimension')
parser.add_argument('--twofold',        type=int,   default=1,                                                      help='is procedure twofold')
parser.add_argument('--load',           type=str,   default=None,                                                   help="location of pre-trained model for twofold procedure")

parser.add_argument('--latent_dims',    type=int,   default=32,                                                     help='number of latent dimensions')
parser.add_argument('--hidden_dims',    type=int,   default=3000,                                                   help='dimension of hidden layers')
parser.add_argument('--hidden_layers',  type=int,   default=3,                                                      help='number of hidden layers')
parser.add_argument('--lr',             type=int,   default=1e-3,                                                   help='learning rate')
parser.add_argument('--epochs',         type=int,   default=4000,                                                   help='number of epochs')
parser.add_argument('--twofold_epochs', type=int,   default=2000,                                                   help='number of twofold epochs')
parser.add_argument('--name',           type=str,   default='perceptive',                                           help='savename')

parser.add_argument('--alpha',           type=float, default=1.0,                                                   help='perceptive regularization weighting')
parser.add_argument('--beta',           type=float, default=1.0,                                                    help='kld regularization weighting')
parser.add_argument('--save_epochs',    type=int,   default=1000,                                                   help='model save periodicity')
parser.add_argument('--saveroot',      type=str,   default=CURRENT_LOCATION+'/trained_models',                      help='results location')

parser.add_argument('--plot_reconstructions', type=int, default=1, help='plot reconstructions while training')
parser.add_argument('--plot_latentspace', type=int, default=1, help='plot latent space while training')
parser.add_argument('--plot_statistics', type=int, default=1, help='plot latent statistics while training')
parser.add_argument('--plot_npoints', type=str, default = 10000, help='number of points used for plotting')

args = parser.parse_args()
args.twofold = bool(args.twofold)
if args.target_dims == 'all':
    args.mds_dims = args.latent_dims

if not args.conditioning is None:
    args.name += "_%s"%args.conditioning
    

#%% Data import

import pickle 

with open('data/trainSet.data','rb') as f:
    audioSet = pickle.load(f)
#testSet = np.load('data/testSet.npy')

if not args.filter is None:
    wrong_ids = np.where(audioSet.metadata['octave'] > args.filter)
    audioSet.files = np.delete(np.array(audioSet.files), wrong_ids).tolist()
    for k, v in audioSet.metadata.items():
        audioSet.metadata[k] = np.delete(v, wrong_ids)

if len(args.frames) == 0:
    print('taking the whole dataset...')
    audioSet.flattenData(lambda x: x[:])
elif len(args.frames)==2:
    print('taking between %d and %d...'%(args.frames[0], args.frames[1]))
    audioSet.flattenData(lambda x: x[args.frames[0]:args.frames[1]])
elif  len(args.frames)==1:
    print('taking frame %d'%(args.frames[0]))
    audioSet.flattenData(lambda x: x[args.frames[0]])

audioSet.constructPartition([], ['train', 'test'], [0.8, 0.2], False)

# Preprocess data
audioSet.data[audioSet.data < 1e-6] = 1e-6
audioSet.data = np.log(audioSet.data)
meanData = np.mean(audioSet.data)
audioSet.data -= meanData
maxData = np.max(np.abs(audioSet.data))
audioSet.data /= maxData

# Define classes
audioSet.classes['instrument'] = { 'English-Horn':0, 'French-Horn':1, 'Tenor-Trombone':2, 'Trumpet-C':3,
                                    'Piano':4, 'Violin':5, 'Violoncello':6, 'Alto-Sax':7, 'Bassoon':8,
                                    'Clarinet-Bb':9, 'Flute':10, 'Oboe':11, '_length':12}
audioSet.classes['pitch'] = {'A':0, 'A#':1, 'B':2, 'C':3, 'C#':4, 'D':5, 'D#':6, 'E':7, 'F':8, 'F#':9, 'G':10, 'G#':11, '_length':12}
audioSet.classes['octave'] = {str(i):i for i in range(9)}
audioSet.classes['octave']['_length'] = 9


#%% Obtain perceptual parameters
timbre_path = 'misc/timbre.npy'
prior_params, prior_gauss_params = percep.get_perceptual_centroids(audioSet, args.mds_dims, timbre_path = timbre_path)

#%% Define model
Model = vaes.VanillaVAE if args.conditioning is None else vaes.ConditionalVAE

if args.load is None:
    input_params = [{'dim':audioSet.data.shape[1], 'dist':dist.Normal}]
    hidden_params = [{'dim':args.hidden_dims, 'nlayers':args.hidden_layers, 'dropout':None, 'residual':False, 'batch_norm':True}]*3
    latent_params = [{'dim':args.latent_dims, 'dist':dist.Normal}]
    add_params = {}
    if not args.conditioning is None:
        add_params['label_params'] = {'dim':audioSet.classes[args.conditioning]['_length'], 'dist':dist.Categorical}
    optim_params = {'optimizer':'Adam', 'optimArgs':{'lr':args.lr}, 'scheduler':'ReduceLROnPlateau'}

    if args.cuda >= 0:
        with torch.cuda.device(args.cuda):
            prior_params = [torch.from_numpy(prior_params[0]).to(torch.float32).cuda(),
                            torch.from_numpy(prior_params[1]).to(torch.float32).cuda()]
            vae = Model(input_params, latent_params, hidden_params, **add_params)
            vae.cuda()
    else:
        prior_params = [torch.from_numpy(prior_params[0]).to(torch.float32),
                        torch.from_numpy(prior_params[1]).to(torch.float32)]
        vae = Model(input_params, latent_params, hidden_params, **add_params)
else:
    a = torch.load(args.load)
    vae = a['class'].load(a)
    if args.cuda >= 0:
        with torch.cuda.device(args.cuda):
            prior_params = [torch.from_numpy(prior_params[0]).to(torch.float32).cuda(),
                            torch.from_numpy(prior_params[1]).to(torch.float32).cuda()]
            vae.cuda()
    else:
        prior_params = [torch.from_numpy(prior_params[0]).to(torch.float32),
                        torch.from_numpy(prior_params[1]).to(torch.float32)]
    


#%% First Training Step

elbo_loss = ELBO({'size_average':False, 'beta':args.beta})
plot_options = {'plot_reconstructions':args.plot_reconstructions, 'plot_statistics':args.plot_statistics, 
                'plot_latentspace':args.plot_latentspace, 'plot_npoints':args.plot_npoints}

if args.twofold and args.load is None:
    n_epochs = args.epochs
    train_options = {'epochs':args.epochs, 'save_epochs':args.save_epochs, 'name':'%s-solo'%args.name, 'results_folder':args.saveroot+'/'+args.name+'-solo'}
    train_model(audioSet, vae, elbo_loss, task=args.conditioning, options=train_options, plot_options=plot_options, save_with={'sh_args':args})


#%% Second Training Step
    
# normalization    
normalize = False

# target regularization dimensions
if args.target_dims in ['min', 'max']:
    out = vae.encode(vae.format_input_data(audioSet.data), 
                      y=vae.format_label_data(audioSet.metadata.get(args.task)))
    zs = [out['z_params_enc'][0][0], out['z_params_enc'][0][1]]
    idx = np.argsort(zs[0].std())
    if (args.targetDims == 'min'):
        targetDims = idx[:args.mds_dims]
    elif (args.targetDims == 'max'):
        targetDims = idx[-args.mds_dims:]
elif args.target_dims == 'first':
    targetDims = np.arange(args.mds_dims)
elif args.target_dims == 'all':
    targetDims = np.arange(args.latent_dims)

# obtain additional criterion
if args.regularization == "l2":
    perceptive_loss = PerceptiveL2Loss(prior_gauss_params[0], targetDims, {'normalize':normalize})
elif args.regularization == "gaussian":
    perceptive_loss = PerceptiveGaussianLoss(prior_gauss_params, targetDims, {'normalize':normalize})
elif args.regularization == "student":
    perceptive_loss = PerceptiveStudent(prior_gauss_params[0], targetDims, {'normalize':normalize})
elif args.regularization == "prior":
    vae.platent = [{'dim':args.latent_dims, 'dist':dist.Normal, 'prior':pg.ClassGaussian(prior_params)}]
if args.regularization in ['l2', 'gaussian', 'student']:
    loss = elbo_loss + args.alpha * perceptive_loss
else:
    loss = elbo_loss


if args.twofold:
    train_options = {'epochs':args.twofold_epochs, 'save_epochs':args.save_epochs, 'name':'%s-reg'%args.name, 'results_folder':args.saveroot+'/'+args.name+'-reg'}
else:
    train_options = {'epochs':args.epochs, 'save_epochs':args.save_epochs, 'name':args.name, 'results_folder':args.saveroot+'/'+args.name}
train_model(audioSet, vae, loss, task=args.conditioning, loss_task = 'instrument', options=train_options, plot_options=plot_options, save_with={'sh_args':args})

