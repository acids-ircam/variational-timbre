# -*- coding: utf-8 -*-
#import re, pdb
import sys, pdb
#import matplotlib
#matplotlib.use('agg')
sys.path.append('../')
import argparse
import librosa
import numpy as np
import torch
import os.path
#from sklearn import manifold
from sklearn.decomposition import PCA
from skimage.transform import resize
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

from nsgt3.cq import NSGT
from nsgt3.fscale import OctScale, MelScale, LogScale
from data.audio import DatasetAudio
import data.metadata as mc
import visualize.visualize_plotting as dr

global meanData
global maxData
global curLatentDims

curLatentDims = 16

"""
###################################
#
# Argument parsing
#
###################################
"""

CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(description='Unsupervised generative timbre spaces')
# Database related arguments
parser.add_argument('--models',         type = str,     nargs = '+',  help = "models to analyze")
parser.add_argument('--output',         type  = str,    default = CURRENT_LOCATION+'/trained_models')
parser.add_argument('--filter', type=int, default = None)

# Representation-related arguments
parser.add_argument('--frames',         type=int, nargs="*", default = [10, 25])
parser.add_argument("--cuda",           type = int,     default = -1)
parser.add_argument('--conditioning',   type = str,     default="pitch")
parser.add_argument('--descriptors_steps', type=int,    default = 60)
# Perform the parsing
parser.add_argument('--plot_reconstructions', type = int, default = 1, help="enables reconstruction plotting")
parser.add_argument('--full_instruments', type = int, default = 1,  help="enables full instruments' sounds reconstruction")
parser.add_argument('--pairwise_paths', type = int, default = 1, help="enables sound sampling from an  instrument distribution to another")
parser.add_argument('--random_paths', type = int, default = 1, help = "enables sound sampling from random paths")
parser.add_argument('--perceptual_infer', type = int, default = 1, help="enables perceptual inference from test dataset")
parser.add_argument('--plot_descriptors', type = int, default = 1, help="enables descriptors video generation")
parser.add_argument('--descriptor_synthesis', type = int, default = 1, help="enables descriptor-based synthesis generation")
parser.add_argument('--labels', type = int, nargs="*", default = [0,4,6], help='in case of conditioning, labels passed for generation')

args = parser.parse_args()

#%%
"""
###################################
#
# [Audio import]
#
# We selected 7,200 samples to represent 12 different instruments with 10 playing styles for each. 
# We normalized the range of notes used by taking common tessitura, and samples annotated with the same intensities 
# Each instrument is represented by 600 samples equally spread across styles.
#
###################################
"""
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

#%%
"""
###################################
#
# [Transform functions]
#
###################################
"""

def scaleData(dataO, direction='forward', scaleType='log'):
    """
    Rescale the data to or from log-amplitude
    """
    if (direction == 'forward'):
        data = dataO.copy()
        data[data < 1e-6] = 1e-6
        data = np.log(data)
        data -= meanData
        data /= maxData
        return data
    else:
        data = dataO.copy()
        data *= maxData
        data += meanData
        data = np.exp(data)
        data[data < 1e-6] = 0
        return data

def regenerateAudio(data, minFreq = 30, maxFreq = 11000, nsgtBins = 48, sr = 22050, scale = 'oct', targetLen = int(3 * 22050), iterations = 100, momentum=False, testSize=False, curName = 'yesssss'):
    # Create a scale
    if (scale == 'oct'):
        scl = OctScale(minFreq, maxFreq, nsgtBins)
    if (scale == 'mel'):
        scl = MelScale(minFreq, maxFreq, nsgtBins)
    if (scale == 'log'):
        scl = LogScale(minFreq, maxFreq, nsgtBins)
    # Create the NSGT object
    nsgt = NSGT(scl, sr, targetLen, real=True, matrixform=True, reducedform=1)
    # Run a forward test 
    if (testSize):
        testForward = np.array(list(nsgt.forward(np.zeros((targetLen)))))
        targetFrames = testForward.shape[1]
        nbFreqs = testForward.shape[0]
        assert(data.shape[0] == nbFreqs)
        assert(data.shape[1] == targetFrames)
    # Now Griffin-Lim dat
    print('Start Griffin-Lim')
    p = 2 * np.pi * np.random.random_sample(data.shape) - np.pi
    for i in range(iterations):
        S = data * np.exp(1j*p)
        inv_p = np.array(list(nsgt.backward(S)))#transformHandler(S, transformType, 'inverse', options)
        new_p = np.array(list(nsgt.forward(inv_p)))#transformHandler(inv_p, transformType, 'forward', options)
        new_p = np.angle(new_p)
        # Momentum-modified Griffin-Lim
        if (momentum):
            p = new_p + ((i > 0) * (0.99 * (new_p - p)))
        else:
            p = new_p
        # Save the output
        librosa.output.write_wav(curName + '.wav', inv_p, sr)

def retrievePathFromDistribution(distrib, vae, cond, pca=None, scale=True, pcaD=3, latentD=curLatentDims):
    if (scale):
        distrib = scaleData(distrib, direction='forward')
    pathLatent = np.zeros((distrib.shape[0], latentD))
    pathPca = np.zeros((1, 1))
    if (pca is not None):
        pathPca = np.zeros((distrib.shape[0], pcaD))
    for f in range(distrib.shape[0]):
        curFrame = distrib[f][np.newaxis, :]
        out = vae.forward(curFrame, y=cond)
        pathLatent[f, :] = out['z_params_enc'][0][0].detach().numpy()
        if (pca is not None):
            pathPca[f, :] = pca.transform(pathLatent[f, :][np.newaxis, :])
    return pathLatent, pathPca

def retrieveDistributionFromPath(path, vae, cond, targetFreq=410, targetFrames=None, pca=None, scale=True):
    # Regenerate distribution from path
    regenDistrib = np.zeros((path.shape[0], targetFreq))
    # Decode 
    for f in range(path.shape[0]):
        decodePoint = (path[f, :][np.newaxis, :])
        if (pca is not None):
            decodePoint = pca.inverse_transform(decodePoint)
        invVal = vae.decode(torch.Tensor(decodePoint), y=cond)
        invVal = invVal[0][0][0].data.numpy()
#        invVal[invVal < 0] = 0
        regenDistrib[f, :] = invVal
    regenDistrib = regenDistrib.T
    if (scale):
        regenDistrib = scaleData(regenDistrib, direction='reverse')
    if (targetFrames is not None):
        regenDistrib = resize(regenDistrib, (regenDistrib.shape[0], targetFrames), mode='constant')
    return regenDistrib

#%%
"""
###################################
#
# [Plotting functions]
#
###################################
"""

def datasetChecking(audioSet, exportName):
    i = 0;
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xVals = np.linspace(0, 1, audioSet.data[0].shape[0])
    line, = ax.plot(xVals, audioSet.data[0])
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    ttl = ax.text(.4, 0.085, os.path.basename(audioSet.files[0]), va='center')
    plt.ylim((-1, 1))
    def updatefig(*args):
        global i
        i += 1
        line.set_data(xVals, audioSet.data[i])
        ttl.set_text(os.path.basename(audioSet.files[i]))
        return line,
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='acids.ircam.fr'), bitrate=1800)
    ani = animation.FuncAnimation(fig, updatefig, frames=audioSet.data.shape[0], interval=50, blit=True)
    ani.save(exportName, writer=writer)
    plt.show()
    plt.close()

def plotReconstructions(dataset=None, vae=None, select='random', nbRows=5, nbColumns=3, figName=None):
    nbExamples = nbRows * nbColumns
    if (select == 'random'):
        ids = np.random.randint(0, dataset.data.shape[0], nbExamples)
    spec = dataset.data[ids]
    metadata = vae.format_label_data(dataset.metadata['pitch'][ids])
    out = vae.forward(spec, y=metadata)['x_params']
    fig, axs = plt.subplots(nbRows, nbColumns * 4, figsize=(18, 8))
    spec_out = out[0][0].data.numpy()
    for i in range(nbRows):
        for j in range(nbColumns):           
            axs[i,j*4].plot(spec[i*nbColumns+j], LineWidth=2)
            axs[i,j*4+1].plot(spec_out[i*nbColumns+j], 'r', LineWidth=2)
            axs[i,j*4+2].plot(scaleData(spec[i*nbColumns+j], direction='reverse'), LineWidth=2)
            axs[i,j*4+3].plot(scaleData(spec_out[i*nbColumns+j], direction='reverse'), 'r', LineWidth=2)
            axs[i,j*4].set_title(os.path.basename(dataset.files[ids[i*nbColumns+j]]).replace('ordinario', '').replace('.wav', ''))
            for tick in axs[i,j].xaxis.get_major_ticks():
                tick.label.set_fontsize('x-small') 
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].tick_params(which='both', labelbottom=False, labelleft=False)
    if (figName is not None):
        plt.savefig(figName, bbox_inches='tight');
        plt.close() 

def plotLatentSpacePath(latentPath=None, dataset=None, vae=None, Zp=None, Mp=None, Zc=None, Mc=None, curTitle='Latent space', figName=None, distribPlot=None):
    pca = None
    s = 10
    if (distribPlot is not None):
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(3, 11) 
        ax = fig.add_subplot(gs[:3,:3], projection='3d')
    else:        
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0], projection='3d')
    if ((dataset is not None) and (vae is not None)):
        y = dataset.metadata['instrument']
        y2 = np.random.randint(0, 11, size=(len(dataset.metadata['instrument'])))
        metadata = vae.format_label_data(dataset.metadata['pitch'])
        out = vae.forward(dataset.data, y=y2)
        Z = out['z_params_enc'][0][0].data.numpy(); zoom = 10
        # Create the PCA output
        s = (torch.exp(torch.mean(out['z_params_enc'][0][1], 1))*zoom).data.numpy()
        pca = PCA(n_components = 3)
        Zp = pca.fit_transform(Z)
        # Create the colors for this
        cmap = dr.get_cmap(np.max(y), color_map='plasma')
        Zc = []
        for i in y:
            color = list(cmap(int(i)))
            color[-1] = 0.5
            Zc.append(color)        
        class_ids, _ = dr.get_class_ids(audioSet, 'instrument')
        Mp = np.array([np.mean(Zp[class_ids[i]], 0) for i in range(len(class_ids))])
        Mc = [cmap(int(i)) for i in range(Mp.shape[0])]
    if (Zp is not None):
        ax.scatter(Zp[:, 0], Zp[:,1], Zp[:, 2], c=Zc, s=s, alpha=0.7)
    if (Mp is not None):
        ax.scatter(Mp[:, 0], Mp[:,1], Mp[:, 2], c=Mc, s=100, alpha=1.0)
    if (latentPath is not None):
        cm = plt.get_cmap('autumn')
        for f in range(latentPath.shape[0]-1):
            ax.plot([latentPath[f, 0], latentPath[f+1, 0]], [latentPath[f, 1], latentPath[f+1, 1]], [latentPath[f, 2], latentPath[f+1, 2]], c=cm(f/latentPath.shape[0]), LineWidth=2)
        ax.scatter(latentPath[0, 0], latentPath[0, 1], latentPath[0, 2], c=cm(0), s=100, alpha=1.0)
        ax.scatter(latentPath[-1, 0], latentPath[-1, 1], latentPath[-1, 2], c=cm(0.9), s=100, alpha=1.0)
    if (distribPlot is not None):
        for i in range(8):
            ax1 = plt.subplot(gs[1, i+3])
            ax1.plot(np.linspace(0, 1, distribPlot[:, 0].shape[0]), distribPlot[:, int(float(i) / 8.0 * (distribPlot.shape[1] - 1))], c=cm(float(i)/8.0))
            ax1.set_ylim([0, np.max(distribPlot)])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.autoscale(enable=True, axis='z', tight=True)
    plt.title(curTitle)
    if (figName is not None):
        plt.savefig(figName, bbox_inches='tight', format='svg');
        plt.close() 
    return pca, Zp, Zc, Mp, Mc

#%%
    
def generatePathVideo(latentPath=None, distribPlot=None, animFrames=40, Zp=None, Mp=None, Zc=None, Mc=None, curTitle='Latent space', figName=None):
    #% Create plot
    fig = plt.figure(figsize=(12, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
    ax = fig.add_subplot(gs[0], projection='3d')
    # Distribution of points
    if (Zp is not None):
        ax.scatter(Zp[:, 0], Zp[:,1], Zp[:, 2], c=Zc, s=10, alpha=0.1)
    if (Mp is not None):
        ax.scatter(Mp[:, 0], Mp[:,1], Mp[:, 2], c=Mc, s=100, alpha=1.0)
    pointLines, = ax.plot(latentPath[0, :2], latentPath[1, :2], latentPath[2, :2])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.autoscale(enable=True, axis='z', tight=True)
    
    #latentPath = latentPath[:, 4:-4]
    
    # Target frames to plot
    global frame
    frame = 0;
    ax1 = plt.subplot(gs[1])
    lines, = ax1.plot(np.linspace(0, 1, distribPlot[:, frame].shape[0]), distribPlot[:, frame])
    ax1.set_ylim([0,np.max(distribPlot)])
    # Function to update
    def updatefig(*args):
        global frame
        frame += 1
        curFrame = int((float(frame) / animFrames)*distribPlot.shape[1])
        if (curFrame >= distribPlot.shape[1]):
            return lines,
        # Pick out a frame from the distribution
        invVal = distribPlot[:, curFrame]
        # Update corresponding plot
        lines.set_data(np.linspace(0, 1, invVal.shape[0]), invVal)
        # Update corresponding plot
        curFrame = int((float(frame) / animFrames)*latentPath.shape[1])
        pointLines.set_data(latentPath[0, :curFrame], latentPath[1, :curFrame])
        pointLines.set_3d_properties(latentPath[2, :curFrame])
        return lines,
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=float(animFrames)/(3), metadata=dict(artist='acids.ircam.fr'), bitrate=1800)
    ani = animation.FuncAnimation(fig, updatefig, frames=animFrames, interval=50, blit=True)
    ani.save(figName + '.mp4', writer=writer)
    ani.event_source.stop()
    del ani
    plt.close()



#%%
"""
###################################
#
# [Combined functions]
#
###################################
"""

def reconstructFullInstruments(dataset, vae, pca, select='loudest', plot=True, video=True, audio=True, baseName='test_', Zp=None, Zc=None, Mp=None, Mc=None):
    # Find the names of instrus
    instruNames = sorted(dataset.classes['instrument'], key=dataset.classes['instrument'].get)
    # Filter out the dataset to only keep loudest elements
    dynamics_ids, _ = dr.get_class_ids(dataset, 'dynamics')
    class_ids, _ = dr.get_class_ids(dataset, 'instrument')
    # Concatenate the loud ids
    loudInstrus = np.concatenate((dynamics_ids[2], dynamics_ids[3]))
    # Now reconstruct all these
    for c1 in range(11):
        print('  - Reconstructing ' + instruNames[c1])
        if (select == 'loudest'):
            class_ids[c1] = np.intersect1d(class_ids[c1], loudInstrus)
        # Select one instrument
        curInstru = class_ids[c1][np.random.randint(len(class_ids[c1]))]
        # Take its full NSGT distribution
        curNSGT = fullData[dataset.revHash[curInstru]];
        # Compute path in latent space
        metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][curInstru]]))
        latentPath, pcaPath = retrievePathFromDistribution(curNSGT, vae, metadata, latentD = vae.platent[-1]['dim'])
        if (plot):
            # Plot this path
            plotLatentSpacePath(latentPath.T, Mp=Mp, Mc=Mc, figName=baseName+str(instruNames[c1]))
        # Compute distribution from this path
        latentPath = vae.format_input_data(latentPath)
        regenDistrib = retrieveDistributionFromPath(latentPath, vae, metadata, targetFrames=targetNSGT25)
        plt.figure('resynth')
        plt.imshow(regenDistrib)
        # Audio this path
        if (audio):
            # Synthesize sound from it
            regenerateAudio(regenDistrib, targetLen = int(2.5*22050), curName=baseName+str(instruNames[c1]))
        if (video):
            curPath = pca.transform(latentPath.detach().numpy()).T
            generatePathVideo(latentPath=curPath, Mp=Mp, Mc=Mc, distribPlot=regenDistrib, figName=baseName+str(instruNames[c1]))
            
            
def reconstructPairwisePaths(dataset, vae, pca, select='note', pathType='spherical', targetFrames=939, expressiveFactor=.7, plot=True, video=True, audio=True, baseName='test_', Zp=None, Zc=None, Mp=None, Mc=None):
    # Find the names of instrus
    instruNames = sorted(dataset.classes['instrument'], key=dataset.classes['instrument'].get)
    # Filter out the dataset to only keep loudest elements
    dynamics_ids, _ = dr.get_class_ids(dataset, 'dynamics')
    class_ids, _ = dr.get_class_ids(dataset, 'instrument')
    # Concatenate the loud ids
    loudInstrus = np.concatenate((dynamics_ids[2], dynamics_ids[3]))
    # Take a note shared amongst all instruments
    pitch_ids, _ = dr.get_class_ids(audioSet, 'pitch')
    curMax, curMaxIDs = 0, 0
    for i in range(len(pitch_ids)):
        if (len(pitch_ids[i]) > curMax):
            curMax = len(pitch_ids[i])
            curMaxIDs = i
    pitch_ids = pitch_ids[curMaxIDs]
    pitchInfos = [None] * 11
    same_pitch_classes = [None] * 11
    # Construct a same pitch class
    for c1 in range(11):
        class_ids[c1] = np.intersect1d(class_ids[c1], loudInstrus)
        pitchInfos[c1] = [None] * (class_ids[c1].shape[0])
        for p in range(len(pitchInfos[c1])):
            pitchInfos[c1][p] = audioSet.metadata['pitch'][class_ids[c1][p]]
        if (c1 == 0):
            finalPitches = np.array(pitchInfos[c1])
        else:
            finalPitches = np.intersect1d(finalPitches, np.array(pitchInfos[c1]))
        same_pitch_classes[c1] = np.intersect1d(class_ids[c1], pitch_ids)
    # Now reconstruct all these
    for c1 in range(11):
        print('    . Start ' + instruNames[c1])
        currentVals = class_ids[c1]
        if (select == 'loudest'):
            currentVals = np.intersect1d(class_ids[c1], loudInstrus)
        if (select == 'note'):
            currentVals = same_pitch_classes[c1]
        # Select one instrument
        curInstru = currentVals[np.random.randint(len(currentVals))]
        # Retrieve starting point
        metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][curInstru]]))
        out = vae.forward(dataset.data[curInstru][np.newaxis, :], y=metadata, volatile=True)
        curStart = out['z_params_enc'][0][0].data.numpy()[0];
        if (pathType == 'expressive'):
            # Take its full NSGT distribution
            curNSGT = fullData[dataset.revHash[curInstru]];
            # Compute path in latent space
            metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][curInstru]]))
            pathC1, pcaPath = retrievePathFromDistribution(curNSGT, vae, metadata)
            pathC1 = pathC1 - pathC1[20, :]
        for c2 in range(11):
            if (c1 == c2):
                continue
            if (pathType != 'expressive' and c2 <= c1):
                continue
            print('    . End ' + instruNames[c2])
            currentVals = class_ids[c2]
            if (select == 'loudest'):
                currentVals = np.intersect1d(class_ids[c2], loudInstrus)
            if (select == 'note'):
                currentVals = same_pitch_classes[c2]
            curInstru = currentVals[np.random.randint(len(currentVals))]
            # Retrieve starting point
            metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][curInstru]]))
            out = vae.forward(dataset.data[curInstru][np.newaxis, :], y=metadata, volatile=True)
            curEnd = out['z_params_enc'][0][0].data.numpy()[0];
            if (pathType == 'expressive'):
                # Take its full NSGT distribution
                curNSGT = fullData[dataset.revHash[curInstru]];
                # Compute path in latent space
                metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][curInstru]]))
                pathC2, pcaPath = retrievePathFromDistribution(curNSGT, vae, metadata)
                pathC2 = pathC2 - pathC2[20, :]
            # Interpolated path (in PCA space)
            projPoints = np.zeros((3, targetFrames))
            realPoints = np.zeros((curLatentDims, targetFrames))
            if (pathType in ['spherical', 'expressive']):
                # First compute the angle for spherical interpolation
                omega = np.arccos(np.dot(curStart / np.linalg.norm(curStart), curEnd / np.linalg.norm(curEnd)))        
            for f in range(targetFrames):
                ratio = (float(f) / targetFrames)
                if (pathType == 'expressive'):                    
                    ratioExpressive = int(ratio * pathC1.shape[0])
                    ratioExpressiveTarget = int(ratio * pathC2.shape[0])
                curPoint = ((1 - ratio) * (curStart)) + (ratio * curEnd)
                if (pathType == 'spherical'):
                    curPoint = (np.sin((1 - ratio) * omega) / np.sin(omega) * (curStart)) + (np.sin(ratio * omega)/np.sin(omega) * (curEnd))
                if (pathType == 'expressive'):
                    curPoint = (np.sin((1 - ratio) * omega) / np.sin(omega) * (curStart + (pathC1[ratioExpressive, :] * expressiveFactor))) + (np.sin(ratio * omega)/np.sin(omega) * (curEnd + (pathC2[ratioExpressiveTarget, :] * expressiveFactor)))
                realPoints[:, f] = curPoint
                projPoints[:, f] = pca.transform(curPoint[np.newaxis, :])
            # Compute distribution from this path
            regenDistrib = retrieveDistributionFromPath(realPoints.T, vae, metadata, targetFrames=targetFrames)
            # Plot this path
            if (plot):
                plotLatentSpacePath(projPoints.T, Mp=Mp, Mc=Mc, distribPlot = regenDistrib, figName=baseName+select+'_'+pathType+'_'+str(instruNames[c1])+'_'+str(instruNames[c2]))
            # Video this path
            if (video):
                generatePathVideo(latentPath=projPoints, Zp=Zp, Mp=Mp, Zc=Zc, Mc=Mc, distribPlot=regenDistrib, figName=baseName+select+'_'+pathType+'_'+str(instruNames[c1])+'_'+str(instruNames[c2]))
            # Audio this path
            if (audio):
                # Synthesize sound from it
                regenerateAudio(regenDistrib, targetLen = int(2.5*22050), curName=baseName+select+'_'+pathType+'_'+str(instruNames[c1])+'_'+str(instruNames[c2]),  iterations=40)
            
"""
###############
#
# N-D paths generator
#
###############
"""
def findPerpendicular(b):
    a = np.zeros(b.shape[0])
    c1 = 0.0
    for i in range(b.shape[0]):
        s = 1.0
        if (b[i] < 0.0):
            s=-1.0
        if (c1 * s > 0.0):
            s = -s
        a[i] = s * np.random.rand()
        c1 += a[i] * b[i]
    for i in range(b.shape[0]):
        if np.fabs(b[i]) > 1e-10:
            c1 -= a[i]*b[i];
            a[i] = - c1 / b[i]
            break;
    a /= np.linalg.norm(a)
    return a

# Create a N-dimensional circle with
#   - dims      : number of dimensions
#   - center    : center position ()
def generateNDcircle(dims, center, radius, points, revolutions):
    # Evolutive radiuses
    if (len(radius) > 1):
        if (len(radius) == 2):
            radius = np.tile(np.linspace(radius[0], radius[1], points)[:, np.newaxis], (1, dims))
        elif (radius.shape[0] != dims):
            radius = np.tile(radius[:, np.newaxis], (1, dims))
    # First generate random unit N-d vector
    u = np.random.rand(dims)
    # Normalize
    u /= np.linalg.norm(u)
    # Generate perpendicular V vector
    v = findPerpendicular(u)
    # Generate the set of values
    t = np.linspace(0, 2 * np.pi * revolutions, points)[:, np.newaxis]
    # Repeat different data
    centerRep = np.tile(center, (points, 1))
    uRep = np.tile(u, (points, 1))
    vRep = np.tile(v, (points, 1))
    tRep = np.tile(t, (1, dims))
    # Generate our circle
    circle = centerRep + (radius*np.cos(tRep)*uRep) + (radius*np.sin(t)*vRep)
    return circle

def testRandomPaths(dataset, vae, pca, targetFrames=1565, plot=True, video=True, audio=True, baseName='test_', nbRepeats=100, curShape='spiralOut', randomStart=True, Zp=None, Zc=None, Mp=None, Mc=None):
    # Random start ?
    if (randomStart):
        # Random center (still close to 0)
        center = np.random.randn(curLatentDims) * 0.2
        cName = 'random_' + str(np.linalg.norm(center))
    else:
        iVal = np.random.randint(Zp.shape[0])
        metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][iVal]]))
        center = vae.forward(dataset.data[iVal][np.newaxis,:], y=metadata)['z_params_enc'][0][0].data.numpy()
        cName = 'ins_' + os.path.basename(dataset.files[iVal])[:-4]
    # Number of points to generate
    points = targetFrames
    # Number of revolutions
    revolutions = (np.random.rand() + 0.025) * 3
    if curShape == 'circle':
        rVal = ((np.random.rand() * np.max(np.abs(Mp))) + 0.1)
        radius = [rVal, rVal]
    if curShape == 'spiralOut':
        rStart = ((np.random.rand() * .9) + 0.1) * .7
        rEnd = rStart + (np.random.rand() * np.max(np.abs(Mp)))
        radius = [rStart, rEnd]
    if curShape == 'spiralIn':
        rEnd = ((np.random.rand() * .9) + 0.1) * .7
        rStart = rEnd + (np.random.rand() * np.max(np.abs(Mp)))
        radius = [rStart, rEnd]
    # Our current funky path
    circlePath = generateNDcircle(curLatentDims, center, radius, points, revolutions)
    # Select a random conditioning
    if (np.random.rand(1) < 1.5):
        curCond = np.random.randint(0, 11)
        metaCond = vae.format_label_data(np.array([curCond]))
        condName = curCond
    else:
        curCond = (resize(np.random.randint(0, 11, size=(4, 1)) / 11, (points, 1), mode='wrap') * 12).astype('int')
        metaCond = vae.format_label_data(curCond)
        condName = 'pattern'
    curName = baseName + curShape + '_' + cName + '_c' + str(condName) + '_r' + str(revolutions) + '_s' + str(radius[0]) + '_e' + str(radius[1])
    # Projection points
    projPoints = pca.transform(circlePath)
    # Compute distribution from this path
    regenDistrib = retrieveDistributionFromPath(circlePath, vae, metaCond, targetFrames=targetFrames)
    if (audio):
        # Synthesize sound from it
        regenerateAudio(regenDistrib, targetLen = int(5*22050), curName=curName)
    # Plot this path
    if (plot):
        plotLatentSpacePath(projPoints, Mp=Mp, Mc=Mc, distribPlot = regenDistrib, figName=curName+'.png')
    # Video this path
    if (video):
        generatePathVideo(latentPath=projPoints.T, Mp=Mp, Mc=Mc, distribPlot=regenDistrib, figName=curName)

#%%
"""
###################################
#
# [Descriptor space computation]
#
###################################
"""
# Set of descriptors we will analyze
descriptors = ['loudness', 'centroid', 'bandwidth', 'flatness', 'rolloff']
# Helper function to sample, synthesize and analyze a point in space
def sampleCompute(model, zPoint, pca, cond, targetDims=None):
    if (targetDims is not None):
        tmpPoint = torch.zeros(3);
        for t in range(len(targetDims)):
            tmpPoint[targetDims[t]] = zPoint[t]
        zPoint = tmpPoint
    # Compute inverse transform at this point
    invVal = model.decode(torch.Tensor(pca.inverse_transform(zPoint)[np.newaxis, :]), y=cond);
    invVal = scaleData(invVal[0][0][0].data.numpy(), direction='reverse').T
    invVal[invVal < 0] = 0
    # Compute all descriptors
    descValues = {'loudness':librosa.feature.rmse(S = invVal),
                  'centroid':librosa.feature.spectral_centroid(S = invVal),
                  'flatness':librosa.feature.spectral_flatness(S = invVal),
                  'bandwidth':librosa.feature.spectral_bandwidth(S = invVal),
                  'rolloff':librosa.feature.spectral_rolloff(S = invVal)}
    return descValues

def sample2DSpace(vae, pca, cond, nbSamples, nbPlanes, Zp, Zc, figName=None):
    # First find boundaries of the space
    spaceBounds = np.zeros((3, 2))
    for i in range(3):
        spaceBounds[i, 0] = np.min(Zp[:, i])
        spaceBounds[i, 1] = np.max(Zp[:, i])
    # Now construct sampling grids for each axis
    samplingGrids = [None] * 3
    for i in range(3):
        samplingGrids[i] = np.meshgrid(np.linspace(-.9, .9, nbSamples), np.linspace(-.9, .9, nbSamples))
    # Create the set of planes
    planeDims = np.zeros((3, nbPlanes))
    for i in range(3):
        curVals = np.linspace(spaceBounds[i, 0], spaceBounds[i, 1], nbPlanes)
        for p in range(nbPlanes):
            planeDims[i, p] = curVals[p]
    dimNames = ['X', 'Y', 'Z'];
    for dim in range(3):
        print('Dimension ' + str(dim))
        curSampling = samplingGrids[dim]
        resultMatrix = {}
        for d in descriptors:
            resultMatrix[d] = [None] * nbPlanes
            for i in range(nbPlanes):
                resultMatrix[d][i] = np.zeros((nbSamples, nbSamples))
        for plane in range(nbPlanes):
            print('Plane ' + str(plane))
            curPlaneVal = planeDims[dim, plane]
            for x in range(nbSamples):
                for y in range(nbSamples):
                    if (dim == 0):
                        curPoint = [curPlaneVal, curSampling[0][x, y], curSampling[1][x, y]]
                    if (dim == 1):
                        curPoint = [curSampling[0][x, y], curPlaneVal, curSampling[1][x, y]]
                    if (dim == 2):
                        curPoint = [curSampling[0][x, y], curSampling[1][x, y], curPlaneVal]
                    descVals = sampleCompute(vae, torch.Tensor(curPoint), pca, cond, targetDims=[0, 1, 2])
                    for d in descriptors:
                        resultMatrix[d][plane][x, y] = descVals[d]
        plt.figure();
        for dI in range(len(descriptors)):
            d = descriptors[dI]
            for i in range(nbPlanes):
                plt.subplot(len(descriptors), nbPlanes, (dI * nbPlanes) + i + 1)
                plt.imshow(resultMatrix[d][i], interpolation="sinc");
                plt.tick_params(which='both', labelbottom=False, labelleft=False)
                if (i == 0):
                    plt.ylabel(d)   
        #plt.subplots_adjust(bottom=0.2, left=0.01, right=0.05, top=0.25)
        if (figName is not None):
            plt.savefig(figName+'_'+dimNames[dim]+'.png', bbox_inches='tight');
            plt.close()
#%% 3D sampling

def sampleBatchCompute(model, zs, pca, cond, targetDims=None):
    if (targetDims is not None):
        zs = zs[targetDims]
    else:
        zs = pca.inverse_transform(zs)
    # Compute inverse transform at this point
    zs = model.format_input_data(zs)
    invVal = model.decode(zs, y=cond.repeat(zs.shape[0], 1));
    invVal = scaleData(invVal[0][0][0].cpu().detach().numpy(), direction='reverse')
    invVal[invVal < 0] = 0
    # Compute all descriptors
    descValues = {'loudness':np.zeros(zs.shape[0]),
                  'centroid':np.zeros(zs.shape[0]),
                  'flatness':np.zeros(zs.shape[0]),
                  'bandwidth':np.zeros(zs.shape[0]),
                  'rolloff':np.zeros(zs.shape[0])}
    for i in range(invVal.shape[0]):
        currentFrame = np.expand_dims(invVal[i], 1)
        descValues['loudness'][i] = librosa.feature.rmse(S = currentFrame)
        descValues['centroid'][i] = librosa.feature.spectral_centroid(S = currentFrame)[0][0]
        descValues['flatness'][i] = librosa.feature.spectral_flatness(S = currentFrame)[0][0]
        descValues['bandwidth'][i] = librosa.feature.spectral_bandwidth(S = currentFrame)[0][0]
        descValues['rolloff'][i] = librosa.feature.spectral_rolloff(S = currentFrame)[0][0]
    return descValues


def getDescriptorGrid(sampleGrid3D, vae, pca, cond):
    # Resulting sampling tensors
    point_hash = {}
    zs = np.zeros((np.ravel(sampleGrid3D[0]).shape[0], 3))
    current_idx = 0
    for x in range(sampleGrid3D[0].shape[0]):
        for y in range(sampleGrid3D[0].shape[1]):
            for z in range(sampleGrid3D[0].shape[2]):
                curPoint = [sampleGrid3D[0][x,y,z],sampleGrid3D[1][x,y,z],sampleGrid3D[2][x,y,z]]
                zs[current_idx] = np.array(curPoint)
                point_hash[(x,y,z)] = current_idx
                current_idx += 1
                
#    cond = vae.format_label_data(np.ones(zs.shape[0]))
    descVals = sampleBatchCompute(vae, zs, pca, cond)
    
    resultTensor = {}
    for d in descVals.keys():
        resultTensor[d] = np.zeros_like(sampleGrid3D[0])
    
    for x in range(sampleGrid3D[0].shape[0]):
        for y in range(sampleGrid3D[0].shape[1]):
            for z in range(sampleGrid3D[0].shape[2]):
                current_idx = point_hash[x,y,z]
                for d in descVals.keys():
                    resultTensor[d][x,y,z] = descVals[d][current_idx]

    return resultTensor
    
    
    
def sample3DSpace(vae, pca, cond, nbSamples, nbPlanes, Zp, Zc, figName=None, loadFrom=None, saveAs=None, resultTensor=None):
    # Create sampling grid
    samplingGrid3D = np.meshgrid(np.linspace(np.min(Zp[:, 0]), np.max(Zp[:, 0]), nbSamples),
                             np.linspace(np.min(Zp[:, 1]), np.max(Zp[:, 1]), nbSamples),
                             np.linspace(np.min(Zp[:, 2]), np.max(Zp[:, 2]), nbSamples))
    # Resulting sampling tensors
    if not loadFrom is None:
        print('loading from %s...'%loadFrom)
        resultTensor = np.load(loadFrom)[None][0]
    elif (resultTensor is None):
        resultTensor = getDescriptorGrid(samplingGrid3D, vae, pca, cond)
#        for d in descriptors:
#            resultTensor[d] = np.zeros((nbSamples, nbSamples, nbSamples))
#        for x in range(nbSamples):
#            for y in range(nbSamples):
#                for z in range(nbSamples):
#                    curPoint = [samplingGrid3D[0][x,y,z],samplingGrid3D[1][x,y,z],samplingGrid3D[2][x,y,z]]
#                    descVals = sampleCompute(vae, torch.Tensor(curPoint), pca, cond, targetDims=[0, 1, 2])
#                    for d in descriptors:
#                        resultTensor[d][x, y, z] = descVals[d]
    if not saveAs is None:
        print('saving as %s...'%saveAs)
        np.save(saveAs, resultTensor)
        
        
    axNames = ['X', 'Y', 'Z']
    # Sets of planes
    xVals = np.linspace(np.min(Zp[:, 0]), np.max(Zp[:, 0]), nbSamples)
    yVals = np.linspace(np.min(Zp[:, 1]), np.max(Zp[:, 1]), nbSamples)
    zVals = np.linspace(np.min(Zp[:, 2]), np.max(Zp[:, 2]), nbSamples)
    for dim in range(3):
        print('-- dim %d...'%dim)
        # For each descriptor
        for d in descriptors:
            print('descriptos %s...'%d)
            global i; i = 0;
            fig = plt.figure(figsize=(12, 6)) 
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
            ax = fig.add_subplot(gs[0], projection='3d')
            plt.title('Projection ' + axNames[dim] + ' - Spectral ' + d)
            if (dim == 0):
                surfYZ = np.array([[xVals[0], np.min(Zp[:, 1]), np.min(Zp[:, 2])],[xVals[0], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                    [xVals[0], np.max(Zp[:, 1]), np.max(Zp[:, 2])],[xVals[0], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
            if (dim == 1):
                surfYZ = np.array([[np.min(Zp[:, 0]), yVals[0], np.min(Zp[:, 2])],[np.max(Zp[:, 0]), yVals[0], np.min(Zp[:, 2])],
                                    [np.max(Zp[:, 0]), yVals[0], np.max(Zp[:, 2])],[np.min(Zp[:, 0]), yVals[0], np.max(Zp[:, 2])]])
            if (dim == 2):
                surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[0]],[np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[0]],
                                    [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[0]],[np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[0]]])
            #ax.scatter(zLatent[:, 0], zLatent[:, 1], zLatent[:, 2])
            task = 'instrument'
            meta = np.array(audioSet.metadata[task])
            cmap = plt.cm.get_cmap('plasma', audioSet.classes[task]['_length'])
            c = []
            for j in meta:
                c.append(cmap(int(j)))   
            ax.scatter(Zp[:, 0], Zp[:,1], Zp[:, 2], c=Zc)
            lines = [None] * 4
            for j in range(4):
                lines[j], = ax.plot([surfYZ[j, 0], surfYZ[(j+1)%4, 0]], [surfYZ[j, 1], surfYZ[(j+1)%4, 1]], zs=[surfYZ[j, 2], surfYZ[(j+1)%4, 2]], linestyle='--', color='k', linewidth=2)
            for v in range(nbSamples):
                if (dim == 0):
                    surfYZ = np.array([[xVals[v], np.min(Zp[:, 1]), np.min(Zp[:, 2])],[xVals[v], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                        [xVals[v], np.max(Zp[:, 1]), np.max(Zp[:, 2])],[xVals[v], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                if (dim == 1):
                    surfYZ = np.array([[np.min(Zp[:, 0]), yVals[v], np.min(Zp[:, 2])],[np.max(Zp[:, 0]), yVals[v], np.min(Zp[:, 2])],
                                        [np.max(Zp[:, 0]), yVals[v], np.max(Zp[:, 2])],[np.min(Zp[:, 0]), yVals[v], np.max(Zp[:, 2])]])
                if (dim == 2):
                    surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[v]],[np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[v]],
                                        [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[v]],[np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[v]]])
                for j in range(4):
                    ax.plot([surfYZ[j, 0], surfYZ[(j+1)%4, 0]], [surfYZ[j, 1], surfYZ[(j+1)%4, 1]], zs=[surfYZ[j, 2], surfYZ[(j+1)%4, 2]], alpha=0.1, color='g', linewidth=2)
            ax1 = plt.subplot(gs[1])  
            if (dim == 0):
                im = ax1.imshow(resultTensor[d][i], animated=True)
            if (dim == 1):
                im = ax1.imshow(resultTensor[d][:, i, :], animated=True)
            if (dim == 2):
                im = ax1.imshow(resultTensor[d][:, :, i], animated=True)
            # Function to update
            def updatefig(*args):
                global i
                i += 1
                try:
                    if (dim == 0):
                        im.set_array(resultTensor[d][i])
                    if (dim == 1):
                        im.set_array(resultTensor[d][:, i, :])
                    if (dim == 2):
                        im.set_array(resultTensor[d][:, :, i])
                    if (dim == 0):
                        surfYZ = np.array([[xVals[i], np.min(Zp[:, 1]), np.min(Zp[:, 2])],[xVals[i], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                            [xVals[i], np.max(Zp[:, 1]), np.max(Zp[:, 2])],[xVals[i], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                    if (dim == 1):
                        surfYZ = np.array([[np.min(Zp[:, 0]), yVals[i], np.min(Zp[:, 2])],[np.max(Zp[:, 0]), yVals[i], np.min(Zp[:, 2])],
                                            [np.max(Zp[:, 0]), yVals[i], np.max(Zp[:, 2])],[np.min(Zp[:, 0]), yVals[i], np.max(Zp[:, 2])]])
                    if (dim == 2):
                        surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[i]],[np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[i]],
                                            [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[i]],[np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[i]]])
                    for j in range(4):
                        lines[j].set_data([surfYZ[j, 0], surfYZ[(j+1)%4, 0]], [surfYZ[j, 1], surfYZ[(j+1)%4, 1]])
                        lines[j].set_3d_properties([surfYZ[j, 2], surfYZ[(j+1)%4, 2]])
                except:
                    print('pass')
                return im,
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='acids.ircam.fr'), bitrate=1800)
            ani = animation.FuncAnimation(fig, updatefig, frames=nbSamples, interval=50, blit=True)
            ani.save(figName + '_' + d + '_' + axNames[dim] + '.mp4', writer=writer)
            ani.event_source.stop()
            del ani
            plt.close()
    return resultTensor

#%%
"""
###################################
#
# [Descriptor-based synthesis]
#
###################################
"""

def descriptorBasedSynthesis(dataset, vae, cond, pca, samplingGrid3D, descriptorSpace, instruPos, curCurve, descVals, figName='test', nbPoints=20, targetFrames=783, audio=True, plot=True, video=True, curName=None, Mp=None, Mc=None):
    # Find the starting point of the instrument
    pdb.set_trace()
    curPoint = [int(np.round((instruPos[0] + 3) * 5)), int(np.round((instruPos[1] + 3) * 5)), int(np.round((instruPos[1] + 2) * 5))]
#    curPoint = [3,2,1]
    nShift = np.array([-1, 0, 1])
    # Keep the N-d path
    fullRealPath = np.zeros((nbPoints, curLatentDims))
    # Generate distribution
    tmpPoint = instruPos.copy()
    tmpPoint[:3] = [samplingGrid3D[0][curPoint[0], curPoint[1], curPoint[2]], samplingGrid3D[1][curPoint[0], curPoint[1], curPoint[2]], samplingGrid3D[2][curPoint[0], curPoint[1], curPoint[2]]]
    fullRealPath[0] = pca.inverse_transform(tmpPoint)
    # Final curve of descriptor
    resCurve = np.zeros(curCurve.shape)
    resCurve[0] = descVals[curPoint[0], curPoint[1], curPoint[2]]
    for p in range(1, nbPoints):
        print(p, curPoint)
        # Check target descriptor evolution
        curDiff = curCurve[p] - curCurve[p - 1]
        # Extract current neighborhood
        curNeighborhood = descVals[curPoint[0]-1:curPoint[0]+2, :, :][:, curPoint[1]-1:curPoint[1]+2, :][:, :, curPoint[2]-1:curPoint[2]+2].copy()
        # Now differentiate with current point
        curNeighborhood -= descVals[curPoint[0], curPoint[1], curPoint[2]]
        curNeighborhood[1, 1, 1] = 1e9
        # Check next point
        nextPoint = np.unravel_index(np.argmin(np.abs(curNeighborhood - curDiff)), curNeighborhood.shape)
        # Update point
        tmpPoint = curPoint.copy()
        for x in range(3):
            tmpPoint[x] += np.min((np.max((nShift[nextPoint[x]], 1)), 48))
        # Update point
        curPoint = tmpPoint
        resCurve[p] = descVals[curPoint[0], curPoint[1], curPoint[2]]
        # Keep the real 3-d point
        tmpPoint = instruPos.copy()
        tmpPoint[:3] = [samplingGrid3D[0][curPoint[0], curPoint[1], curPoint[2]], samplingGrid3D[1][curPoint[0], curPoint[1], curPoint[2]], samplingGrid3D[2][curPoint[0], curPoint[1], curPoint[2]]]
        fullRealPath[p] = pca.inverse_transform(tmpPoint)
    tmpPoint = instruPos.copy()
    tmpPoint[:3] = [samplingGrid3D[0][curPoint[0], curPoint[1], curPoint[2]], samplingGrid3D[1][curPoint[0], curPoint[1], curPoint[2]], samplingGrid3D[2][curPoint[0], curPoint[1], curPoint[2]]]
    fullRealPath[nbPoints-1] = pca.inverse_transform(tmpPoint)
    upsamplePath = np.zeros((nbPoints * 20, curLatentDims))
    # Now upsample the distribution
    for f in range(nbPoints * 20):
        ratio = (f % 20)
        idP = int(np.floor(f / 20))
        tmpPoint = ((1 - ratio) * fullRealPath[idP]) + ratio * fullRealPath[min(idP + 1, nbPoints-1)]
        upsamplePath[f, :] = tmpPoint
    # Compute distribution from this path
    regenDistrib = retrieveDistributionFromPath(upsamplePath, vae, cond, targetFrames=targetFrames)
    if (audio):
        # Synthesize sound from it
        regenerateAudio(regenDistrib, targetLen = int(2.5*22050), curName=curName)
    # Plot this path
    if (plot):
        plotLatentSpacePath(upsamplePath, Mp=Mp, Mc=Mc, distribPlot = regenDistrib, figName=curName+'.png')
    # Video this path
    if (video):
        generatePathVideo(latentPath=upsamplePath.T, Mp=Mp, Mc=Mc, distribPlot=regenDistrib, figName=curName)

    
def descriptorBasedTest(dataset, vae, pca, Zp, cond, select='loudest', nbRepeat = 20, nbPoints = 20, descriptorSpace=None, baseName=None, Mp=None, Mc=None):
    # Create sampling grid
    samplingGrid3D = np.meshgrid(np.linspace(np.min(Zp[:, 0]), np.max(Zp[:, 0]), descriptorSpace['centroid'].shape[0]),
                             np.linspace(np.min(Zp[:, 1]), np.max(Zp[:, 1]), descriptorSpace['centroid'].shape[0]),
                             np.linspace(np.min(Zp[:, 2]), np.max(Zp[:, 2]), descriptorSpace['centroid'].shape[0]))
    # Define some evolution curves
    linearUp = np.linspace(0, 1, nbPoints)
    linearDown = -linearUp
    expUp = np.exp(linearUp)
    expDown = -expUp
    logUp = np.log(linearUp + 1)
    logDown = -logUp
    curvesTarget = [linearUp, linearDown, expUp, expDown, logUp, logDown]
    curveNames = ['linearUp', 'linearDown', 'expUp', 'expDown', 'logUp', 'logDown']
    for c in range(len(curvesTarget)):
        curvesTarget[c] /= np.max(np.abs(curvesTarget[c]))
    # Selection of IDs
    selectIDs = np.linspace(0, dataset.data.shape[0]-1, dataset.data.shape[0])
    if (select == 'loudest'):
        dynamics_ids, _ = dr.get_class_ids(audioSet, 'dynamics')
        selectIDs = np.concatenate((dynamics_ids[2], dynamics_ids[3]))
    # Selected descriptor
    for desc in ['centroid', 'flatness', 'bandwidth']:
        descVals = descriptorSpace[desc].copy()
        # Normalize descriptor
        descVals /= np.max(descVals)
        for repeat in range(nbRepeat):
            # Select a random point in the dataset
            sInstru = selectIDs[np.random.randint(len(selectIDs))]
            # Find its latent position
            metadata = vae.format_label_data(np.array([dataset.metadata['pitch'][sInstru]]))
            out = vae.forward(dataset.data[sInstru][np.newaxis, :], metadata)
            instruPos = pca.transform(out['z_params_enc'][0][0].data.numpy())[0]
            # Now try out 3 different shapes
            for cIDtest in range(3):
                cID = np.random.randint(len(curvesTarget))
                curCurve = curvesTarget[cID]
                curName = baseName + os.path.basename(dataset.files[sInstru])[:-4] + '_' + desc + '_' + curveNames[cID]
                descriptorBasedSynthesis(dataset, vae, cond, pca, samplingGrid3D, descriptorSpace, instruPos, curCurve, descVals, curName=curName, Mp=Mp, Mc=Mc)
    
                
                
#%%
"""
###################################
#
# [Novel dataset projection]
#
###################################
"""


def projectDataset(setRoot, vae, pca, Zp, Zc, Mp, Mc, figName=None):
    # First import the dataset
#    testSet = importTestDataset('/Users/chemla/Datasets/acidsInstruments-test', '/Users/chemla/Datasets/acidsInstruments-test/analysis/ordinario')
    global testSet
    # Collect properties
    metadata = testSet.metadata['pitch']
    # Encode current data
    data = vae.format_input_data(testSet.data)
    metadata = vae.format_label_data(metadata)
    with torch.no_grad():
        out = vae.forward(data, y=metadata)
        
    Ztest = out['z_params_enc'][0][0].data.numpy()
    #s = (torch.exp(torch.mean(out['z_params_enc'][0][1], 1))*10).data.numpy()
    # Now use the training PCA
    Ztest = pca.transform(Ztest)
    # Retrieve instrument names
    
    import matplotlib.patches as mpatches
    train_classes = audioSet.classes.get('instrument')
    test_classes = testSet.classes.get('instrument')
    test_hash = {k: v  for v, k in test_classes.items() }
    cmap_test = dr.get_cmap(test_classes['_length'], color_map='viridis')
    has  = {}
    
    train_handles = []
    for k,v in train_classes.items():
        if not k == '_length':
            has[v] = k
            print(k,v)
            patch = mpatches.Patch(color=Mc[v], label=k)
            train_handles.append(patch)
            
    test_handles = []
    for k,v in test_classes.items():
        if not k == '_length':
            has[v] = k
            patch = mpatches.Patch(color=cmap_test(v), label=k)
            test_handles.append(patch)
    
    # Compute centroid positions for classes
    class_ids, class_names = dr.get_class_ids(testSet, 'instrument')
    Mtest = np.array([np.mean(Zp[class_ids[i]], 0) for i in range(len(class_ids))])
    Mtc = [cmap_test(int(i)) for i in range(Mtest.shape[0])]
    # First plot only centroids of both
    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    ax.scatter(Mp[:, 0], Mp[:,1], Mp[:, 2], c=Mc, s=100)
    ax.scatter(Mtest[:, 0], Mtest[:,1], Mtest[:, 2], c=Mtc, s=100)
    for i, test_class in enumerate(class_names):
        class_name = test_hash[test_class]
        ax.text(Mtest[i, 0] + 0.01, Mtest[i,1]+0.01, Mtest[i, 2]+0.01, class_name, color=cmap_test(i))
    ax.legend(train_handles, loc=1, bbox_to_anchor=(1.5, 1.0))
#    pdb.set_trace()
    if (figName is not None):
        fig.savefig(figName +'space_centroid.svg', bbox_inches='tight', format='svg');
        plt.close()     
    # Plot full test set with original centroids
    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    ax.scatter(Mp[:, 0], Mp[:,1], Mp[:, 2], c=Mc, s=100)
#    pdb.set_trace()
    ax.scatter(Ztest[:, 0], Ztest[:,1], Ztest[:, 2], c=cmap_test(testSet.metadata.get('instrument')), s=10)
    ax.legend(handles=test_handles, loc=1, bbox_to_anchor=(1.5, 1.0))
    if (figName is not None):
        plt.savefig(figName +'space_mixed.png', bbox_inches='tight', format='svg');
        plt.close()     
    # Now plot all points together
    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    ax.scatter(Zp[:, 0], Zp[:,1], Zp[:, 2], c=Zc, s=10, alpha=0.3)
    ax.scatter(Ztest[:, 0], Ztest[:,1], Ztest[:, 2], c=cmap_test(testSet.metadata.get('instrument')), s=10)
    ax.legend(handles=train_handles  + test_handles, loc=1, bbox_to_anchor=(1.5, 1.0))

    if (figName is not None):
        plt.savefig(figName +'space_all.png', bbox_inches='tight', format='svg');
        plt.close()     
    # Plot some random reconstructions of this set
    for repeat in range(3):
        plotReconstructions(vae = vae, dataset = testSet, figName=figName+'reconstructions_'+str(repeat)+'.png')

#projectDataset('/Users/chemla/Datasets/acidsInstruments-test', vae, pca, Zp, Zc, Mp, Mc, figName=curDir+'novel_projection/')

#%%
"""
###################################
#
# [Testing various models]
#
###################################
"""

# Sets of target sizes for NSGT
global targetNSGT25;    targetNSGT25    = 783
global targetNSGT3;     targetNSGT3     = 939
global targetNSGT35;    targetNSGT35    = 1096
global targetNSGT5;     targetNSGT5     = 1565


models = args.models
modelName = [os.path.splitext(x)[0] for  x in models]
baseDir = args.output

#%%       
        
for mID in range(len(models)):
    modelDir = os.path.dirname(os.path.abspath(models[mID]))
    model = models[mID]
    name = modelName[mID]
    curDir = baseDir + '/' + name + '/'
    if (not os.path.exists(curDir)):
        os.makedirs(curDir)
        for r in range(3):
            os.makedirs(curDir + 'recons_full_%d/'%r)
        os.makedirs(curDir + 'recons_pairwise/')
        os.makedirs(curDir + 'random_paths/')
        os.makedirs(curDir + 'descriptor_space/')
        os.makedirs(curDir + 'descriptor_synthesis/')
        os.makedirs(curDir + 'novel_projection/')
    print('Checking ' + name)
    # Load current model
    try:
        with torch.cuda.device(args.cuda):
            a = torch.load(model)
            vae = a["class"].load(a)
    except AttributeError as e:
        print(e)
        continue
    # Put in evaluation mode
    vae.eval()
    print('  - Plot the latent space')
    pca, Zp, Zc, Mp, Mc = plotLatentSpacePath(vae = vae, dataset = audioSet, figName=curDir+'spacePCA.png')
    for r in range(2):
        if args.plot_reconstructions == 1:
            print('  - Plot some random reconstructions inside the set')
            plotReconstructions(vae = vae, dataset = audioSet, figName=curDir+'reconstructions_'+str(r)+'.png')
        if args.full_instruments == 1:
            print('  - Reconstruct full instruments')
            reconstructFullInstruments(audioSet, vae, baseName=curDir + 'recons_full_%d/'%r, Zp=Zp, Zc=Zc, Mp=Mp, Mc=Mc)
    
    if args.pairwise_paths == 1:
        print('  - Reconstruct pairwise paths')
        reconstructPairwisePaths(audioSet, vae, pca, pathType='spherical', targetFrames=targetNSGT25, baseName=curDir + 'recons_pairwise/', Zp=Zp, Zc=Zc, Mp=Mp, Mc=Mc)    
        reconstructPairwisePaths(audioSet, vae, pca, pathType='expressive', targetFrames=targetNSGT25, baseName=curDir + 'recons_pairwise/', Zp=Zp, Zc=Zc, Mp=Mp, Mc=Mc)    
        reconstructPairwisePaths(audioSet, vae, pca, select='loudest', pathType='expressive', targetFrames=targetNSGT25, baseName=curDir + 'recons_pairwise/', Zp=Zp, Zc=Zc, Mp=Mp, Mc=Mc)
    
    if args.random_paths == 1:
        print('  - Test random paths')
        for repeat in range(3):
            for shape in ['spiralOut', 'spiralIn']:
                for start in [True, False]:
                    testRandomPaths(audioSet, vae, pca, targetFrames=targetNSGT5, plot=True, baseName=curDir+'random_paths/', curShape=shape, randomStart=start, Zp=Zp, Zc=Zc, Mp=Mp, Mc=Mc)
    
    # Compute a full PCA
    if args.perceputal_infer == 1 or args.plot_descriptor == 1 or args.descriptor_synthesis == 1:
        Z = vae.forward(vae.format_input_data(audioSet.data), y=vae.format_label_data(audioSet.metadata['pitch']))['z_params_enc'][0][0].data.numpy()
        pca_f = PCA(n_components = 3)
        Zp_f = pca_f.fit_transform(Z)
        if args.perceputal_infer == 1:
            print('  - Projecting a new dataset')
            projectDataset('/Users/chemla/Datasets/acidsInstruments-test', vae, pca, Zp, Zc, Mp, Mc, figName=curDir+'novel_projection/')
        
        if args.plot_descriptors == 1 or args.descriptor_synthesis == 1:
            print('  - Create 2D descriptor space')       
            for curCond in args.labels:
                sampleGridPath = baseDir + '/' + modelName[mID]+'/descriptor_space/'+ modelName[mID]+'_sampleGrid_'+str(args.descriptors_steps)+'_'+str(curCond)+'.npy'
                loadFrom = sampleGridPath if os.path.isfile(sampleGridPath) else None
                saveAs = sampleGridPath if not os.path.isfile(sampleGridPath) else None
                
                metaCond = vae.format_label_data(np.array([curCond]))
        #        sample2DSpace(vae, pca_f, metaCond, 10, 3, Zp, Zc, figName=curDir+'descriptor_space/space2D_c'+str(curCond))
                print('  - Create 3D descriptor space')
                resultTensor = sample3DSpace(vae, pca_f, metaCond, args.descriptors_steps, 10, Zp, Zc, loadFrom=loadFrom, saveAs=saveAs, figName=curDir+'descriptor_space/space3D_c'+str(curCond))
                
                if args.descriptor_synthesis == 1:
                    print('  - Performing descriptor-based synthesis')
                    descriptorBasedTest(audioSet, vae, pca_f, Zp_f, metaCond, select='loudest', nbPoints = 20, descriptorSpace=resultTensor, baseName=curDir+'descriptor_synthesis/', Mp=Mp, Mc=Mc)
