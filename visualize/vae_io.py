#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:55:58 2017

@author: chemla
"""
import matplotlib.pyplot as plt
from .dimension_reduction import *
import numpy as np
try:
    import imageio         
except:
    pass
from sklearn.metrics import confusion_matrix
from utils import fromOneHot
import os.path
import librosa, itertools
from torch.autograd import Variable

######################################################################################
######
####       General methods
    

def plot_confusion_matrix(cm, classes,
                          
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_confusion_matrix(out, dataset, task):
    cnf_matrix = confusion_matrix(fromOneHot(out['y'].data.numpy()), fromOneHot(dataset.metadata[task]))
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[str(x) for x in range(10)],
                          title='Confusion matrix, without normalization')


def make_grid(images, shape, original_shape):
    if images.shape[0] != shape[0]*shape[1]:
        raise Exception("Number of images %d does not match the given shape : %s"%(images.shape[0], shape))
    grid = np.zeros((shape[0]*original_shape[0], shape[1]*original_shape[1]))
    idx=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid[i*original_shape[0]:(i+1)*original_shape[0], j*original_shape[1]:(j+1)*original_shape[1]] = np.reshape(images[idx, :], original_shape)
            idx += 1
    return np.reshape(grid, (1, grid.shape[0], grid.shape[1]))





######################################################################################
######
####        Image-related methods
    

def export_img(path, model, z, original_size, y=None, options={}):
    original_size = original_size or None
    x_params, _, _ = model.decode(z, y=y)
    x = model.pinput[0]['dist'].analytic_mean(*x_params[0])
    x = np.reshape(x.data.numpy(), original_size)
    imageio.imwrite(path, x)
    

def export_pairs_img(path, dataset, model, ids=None, task=None, original_size=None):
    original_size = original_size or dataset.original_size or None
    if original_size is None:
        raise Exception('original size of image not found in arguments nor dataset attributes.')
    if not issubclass(type(ids), list) and type(ids)!=np.ndarray:
        if type(ids) == int or type(ids) == float:
            ids = np.random.permutation(len(dataset.data))[:int(ids)]
        else:
            raise Exception('ids key argument seems to be uncorrect. Please give an array of ids or a number of images to draw.')
    data = dataset.data[ids]
    if not task is None:
        metadata = dataset.metadata[task][ids]
    else:
        metadata = None
    out = model.forward(data, y=metadata, volatile=True)
    x_out = out['x_params'][0][0].data
    if model.useCuda:
        x_out = x_out.cpu()
    x_out = x_out.numpy()
    al = np.concatenate((data, x_out), 0)
    grid = make_grid(al, (2, len(ids)), original_size)
    imageio.imwrite(path, np.transpose(grid, (1,2,0)))
    
def export_grid_img(path, model, original_shape, sample_range=[-3, 3], n_steps=10, manifold_name=None, layer=0, y_label=None):
    latent_dim = model.platent[layer]['dim']
    x = np.linspace(sample_range[0], sample_range[1], n_steps)
    y = np.linspace(sample_range[0], sample_range[1], n_steps)
    z = np.zeros((n_steps**2, 2))
    X,Y = np.meshgrid(x,y)
    iterator = np.nditer(X, flags=['multi_index'])
    index= 0
    for i in iterator:
        mindex = iterator.multi_index
        z[index, :] = np.array([X[mindex], Y[mindex]])
        index += 1        
        
    if latent_dim > 2:
        if manifold_name is None:
            raise Exception('latent space dimension is greater than 2 ; please provide a adequate manifold')
        try:
            manifold = model.manifolds[manifold_name]
        except:
            raise Exception('model does not seem to possess the manifold %s. Please give a valid manifold name'%manifold)
        print("[export_img_grid]-- dim(z) > 2. processing to dimension reduction with manifold %s..."%manifold_name)
        z = manifold.inverse_transform(z)
    
    z = Variable(torch.from_numpy(z.astype('float32')), volatile=True)
    if model.useCuda:
        z = z.cuda()
    z = [z]
    y = model.format_label_data(y_label, volatile=True)
    x_params, _, _ = model.decode(z, y=y)
    grid = make_grid(x_params[0][0].data.numpy(), (n_steps, n_steps), original_shape)
    imageio.imwrite(path, np.transpose(grid, (1,2,0)))






######################################################################################
######
####        Audio plotting methods


def plot_reconstructions(model, dataset, n_samples=5, ids=None):
    if ids is None:
        n_batches = dataset.data.shape[0] if not issubclass(type(dataset.data[0]), list) else dataset.data[0].shape[0]
        spec_id = np.random.choice(list(range(n_batches)), size=n_samples**2)
    else:
        spec_id = ids
    if issubclass(type(dataset.data), list):
        spec = [x[spec_id] for x in dataset.data]
    else:
        spec = [dataset.data[spec_id]]
    out = model.forward(model.format_input_data(spec))['x_params']
    fig, axs = plt.subplots(n_samples, n_samples)
    for s in range(len(out)):
        spec_out = out[s][0].to('cpu').detach().numpy()
        for i in range(n_samples):
           for j in range(n_samples):
               axs[i,j].plot(spec[s][i*n_samples+j], LineWidth=0.5)
               spec_out[s][i*n_samples+j] /= np.max(spec_out[s][i*n_samples+j])
               axs[i,j].plot(spec_out[i*n_samples+j], LineWidth=0.5)
               for tick in axs[i,j].xaxis.get_major_ticks():
                    tick.label.set_fontsize('x-small') 
               for tick in axs[i,j].yaxis.get_major_ticks():
                    tick.label.set_fontsize('x-small') 
    return fig, axs






######################################################################################
######
####        Synthesis methods

#
#from ..data.sets.signal.transforms import transformHandler, inverseTransform
#
#def export_resynthesis(audioList, model, transformOptions, method='griffin_lim', normalize=True, transform_id=0, *args, **kwargs):
#    if not issubclass(type(audioList), list):
#        audioList = [audioList]
#    outs = []
#    for audioFile in audioList:
#        head, tail = os.path.split(audioFile)
#        name,_ = os.path.splitext(tail)
#        sig, sr = librosa.load(audioFile)
#        original_phase = None
#        if method=="originalPhase":
#            original_phase = np.angle(librosa.stft(sig, transformOptions['winSize'], transformOptions['hopSize'])).T
#        currentTransform = transformHandler(sig, transformOptions['transformTypes'], 'forward', transformOptions)
#        out = model.forward(currentTransform, *args, **kwargs)
#        outs.append(out)
#        sig_out = model.pinput[0]['dist'].analytic_mean(*out['x_params'][0])
#        sig_out = sig_out.data.numpy()
#        sig_out = inverseTransform(sig_out, transformOptions['transformTypes'][transform_id], {'transformOptions':transformOptions['transformParameters']}, 
#                                   method=method, originalPhase=original_phase, *args, **kwargs)
#        new_name = head+name+'_resynth.wav'
#        
#        if normalize:
#            rms = np.mean(sig_out**2)
#            sig_out = sig_out / np.sqrt(rms)
#        
#        librosa.output.write_wav(new_name, sig_out, sr)
#        
#    return outs
#
#def export_sample(path, z, model, transformOptions, repeat=1, transform_id=0 ,*args, **kwargs):
#    if not os.path.isdir(path):
#        os.makedirs(path)
#    x_params, _, _ = model.decode(z, *args, **kwargs)
#    x = model.pinput[0]['dist'].analytic_mean(*x_params[0])
#    for i in range(x.shape[0]):
#        currentTransform = x[i, :].view(1, x.size(1)).repeat(repeat, 1)
#        currentTransform  = currentTransform.data.numpy()
#        options = transformOptions['transformParameters'][0]
#        print(transformOptions['transformParameters'])
#        sig_out = inverseTransform(currentTransform, transformOptions['transformTypes'][transform_id], options, *args, **kwargs)
#        librosa.output.write_wav(path+'/'+'resyn_%d.wav'%i, sig_out, transformOptions['resampleTo'])
