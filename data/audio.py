"""

 Import toolbox       : Audio dataset import

 This file contains the definition of an audio dataset import.

 Author               : Philippe Esling
                        <esling@ircam.fr>

"""

try:
    from matplotlib import pyplot as plt
except:
    import matplotlib 
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    
    
import numpy as np
import scipy as sp
import os
import math
import re

# Package-specific import
from . import utils as du
from . import generic
#from .signal.transforms import computeTransform
from librosa import load as lbload

"""
###################################
# Initialization functions
###################################
"""

class DatasetAudio(generic.Dataset):    
    """ Definition of a basic dataset
    Attributes:
        dataDirectory: 
    """
        
    def __init__(self, options):
        super(DatasetAudio, self).__init__(options)
        # Accepted types of files
        self.types = options.get("types") or ['mp3', 'wav', 'wave', 'aif', 'aiff', 'au'];
        self.importBatchSize = options.get("importBatchSize") or 64;
        self.transformName = options.get("transformName") or None;
        self.matlabCommand = options.get("matlabCommand") or '/Applications/MATLAB_R2014b.app/bin/matlab';
        self.forceRecompute = options.get("forceRecompute") or False;
        self.backend = options.get("backend") or "python"
        # Type of audio-related augmentations
        self.augmentationCallbacks = [];
        
    """
    ###################################
    # Transform functions
    ###################################
    """
    
    def computeTransforms(self, idList, options, padding=False):
        raise NotImplementedError()
        self.dataPrefix = options.get('dataPrefix') or self.dataPrefix
        self.dataDirectory = options.get('dataDirectory') or self.dataDirectory or self.dataPrefix+'/data'
        self.analysisDirectory = options.get('analysisDirectory') or self.analysisDirectory or self.dataPrefix+'/analysis'
        transformParameters = options.get('transformParameters') or [self.getTransforms()[1]]
        transformNames = options.get('transformNames') or ['stft']
        transformTypes = options.get('transformTypes') or ['stft']
        forceRecompute = options.get('forceRecompute') or False
        verbose = options.get('verbose') or False
        if len(transformNames)!=len(transformTypes):
            raise Exception('please give the same number of transforms and names')
        # get indices to compute
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            if padding:
                indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
                indices = np.split(indices, len(indices) / self.importBatchSize)
            else:
                indices = [indices]
        else:
            indices = np.array(idList)
        if (len(self.data) == 0):
            self.data = [None] * len(self.files)
            
        if not os.path.isdir(self.analysisDirectory):
            os.makedirs(self.analysisDirectory)
        
        
        for v in indices:
            curFiles = [None] * v.shape[0]
            for f in range(v.shape[0]):
                curFiles[f] = self.files[int(v[f])]
                
            for i in range(len(transformTypes)):
                analysis_dir = self.analysisDirectory+'/'+transformNames[i]
                if not os.path.isdir(analysis_dir):
                    os.makedirs(analysis_dir)
                transformOptions = {'dataDirectory':self.dataDirectory,
                                    'dataPrefix':self.dataPrefix, 
                                    'analysisDirectory':analysis_dir,
                                    'transformParameters':transformParameters[i],
                                    'transformType':[transformTypes[i]], 
                                    'forceRecompute':forceRecompute,
                                    'verbose':verbose}
                makeAnalysisFiles(curFiles, transformOptions)
                np.save(analysis_dir+'/transformOptions.npy', transformOptions)

    """
    ###################################
    # Import functions
    ###################################
    """
    
    def importData(self, idList, options, padding=False):
        """ Import part of the audio dataset (linear fashion) """
        options["matlabCommand"] = options.get("matlabCommand") or self.matlabCommand;
        options["transformName"] = options.get("transformName") or self.transformName;
        options["dataDirectory"] = options.get("dataDirectory") or self.dataDirectory;
        options["analysisDirectory"] = options.get("analysisDirectory") or self.analysisDirectory;
        
        # We will create batches of data
        indices = []
        # If no idList is given then import all !
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            if padding:
                indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
                indices = np.split(indices, len(indices) / self.importBatchSize)
            else:
                indices = [indices]
        else:
            indices = np.array(idList)
        # Init data
        self.data = [None] * len(self.files)
        
        # Parse through the set of batches
        for v in indices:
            curFiles = [None] * v.shape[0]
            for f in range(v.shape[0]):
                curFiles[f] = self.files[int(v[f])]
            curData, curMeta = importAudioData(curFiles, options)
            for f in range(v.shape[0]):
                self.data[int(v[f])] = curData[f]
        
    def importRawData(self, grainSize=512, overlap = 2, ulaw=256):
        self.data = []
        for f in self.files:
            sig, sr = lbload(f)
            if ulaw > 2 :
                sig = signal.MuLawEncoding(ulaw)(sig)
            n_chunks = sig.shape[0] // (grainSize // overlap) - 1
            grainMatrix = np.zeros((n_chunks, grainSize))
            for i in range(n_chunks):
                idxs = (i*(grainSize // overlap), i*(grainSize // overlap)+grainSize)
                grainMatrix[i] = sig[idxs[0]:idxs[1]]
            self.data.append(grainMatrix)

    """
    ###################################
    # Get asynchronous pointer and options to import
    ###################################
    """
    def getAsynchronousImport(self):
        a, transformOpt = self.getTransforms()
        options = {
            "matlabCommand":self.matlabCommand,
            "transformType":self.transformType,
            "dataDirectory":self.dataDirectory,
            "analysisDirectory":self.analysisDirectory,
            "forceRecompute":self.forceRecompute,
            "transformOptions":transformOpt,
            "backend":self.backend,
            "verbose":self.verbose 
            }
        return importAudioData, options


    """
    ###################################
    # Obtaining transform set and options
    ###################################
    """
    
    def getTransforms(self):
        """
        Transforms (and corresponding options) available
        """
        # List of available transforms
        transformList = [
            'raw',                # raw waveform
            'stft',               # Short-Term Fourier Transform
            'mel',                # Log-amplitude Mel spectrogram
            'mfcc',               # Mel-Frequency Cepstral Coefficient
#            'gabor',              # Gabor features
            'chroma',             # Chromagram
            'cqt',                # Constant-Q Transform
            'gammatone',          # Gammatone spectrum
            'dct',                # Discrete Cosine Transform
#            'hartley',            # Hartley transform
#            'rasta',              # Rasta features
#            'plp',                # PLP features
#            'wavelet',            # Wavelet transform
#            'scattering',         # Scattering transform
#            'cochleogram',        # Cochleogram
            'strf',               # Spectro-Temporal Receptive Fields
            'csft',                 # Cumulative Sampling Frequency Transform
            'modulation',          # Modulation spectrum
            'nsgt',               # Non-stationary Gabor Transform
            'nsgt-cqt',               # Non-stationary Gabor Transform (CQT scale)
            'nsgt-mel',               # Non-stationary Gabor Transform (Mel scale)
            'nsgt-erb',               # Non-stationary Gabor Transform (Mel scale)
            'strf-nsgt',              # Non-stationary Gabor Transform (STRF scale)
        ];
                
        # List of options
        transformOptions = {
            "debugMode":0,
            "resampleTo":22050,
            "targetDuration":0,
            "winSize":2048,
            "hopSize":1024,
            #"nFFT":2048,
            # Normalization
            "normalizeInput":False,
            "normalizeOutput":False,
            "equalizeHistogram":False,
            "logAmplitude":False,
            #Raw features
            "grainSize":512,
            "grainHop":512,
            # Phase
            "removePhase":False,
            "concatenatePhase":False,
            # Mel-spectrogram
            "minFreq":30,
            "maxFreq":11000,
            "nbBands":128,
            # Mfcc
            "nbCoeffs":13,
            "delta":0,
            "dDelta":0,
            # Gabor features
            "omegaMax":'[pi/2, pi/2]',
            "sizeMax":'[3*nbBands, 40]',
            "nu":'[3.5, 3.5]',
            "filterDistance":'[0.3, 0.2]',
            "filterPhases":'{[0, 0], [0, pi/2], [pi/2, 0], [pi/2, pi/2]}',
            # Chroma
            "chromaWinSize":2048,
            # CQT
            "cqtBins":48,
            "cqtFreqMin":64,
            "cqtFreqMax":8000,
            "cqtGamma":0.5,
            # Gammatone
            "gammatoneBins":64,
            "gammatoneMin":64,
            # Wavelet
            "waveletType":'\'gabor_1d\'',
            "waveletQ":8,
            # Scattering
            "scatteringDefault":1,
            "scatteringTypes":'{\'gabor_1d\', \'morlet_1d\', \'morlet_1d\'}',
            "scatteringQ":'[8, 2, 1]',
            "scatteringT":8192,
            # Cochleogram
            "cochleogramFrame":64,        # Frame length, typically, 8, 16 or 2^[natural #] ms.
            "cochleogramTC":16,           # Time const. (4, 16, or 64 ms), if tc == 0, the leaky integration turns to short-term avg.
            "cochleogramFac":-1,          # Nonlinear factor (typically, .1 with [0 full compression] and [-1 half-wave rectifier]
            "cochleogramShift":0,         # Shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
            "cochleogramFilter":'\'p\'',      # Filter type ('p' = Powen's IIR, 'p_o':steeper group delay)
            # STRF
            "strfFullT":0,                # Fullness of temporal margin in [0, 1].
            "strfFullX":0,                # Fullness of spectral margin in [0, 1].
            "strfBP":0,                   # Pure Band-Pass indicator
            "strfRv": np.power(2, np.linspace(0, 5, 5)),     # rv: rate vector in Hz, e.g., 2.^(1:.5:5).
            "strfSv": np.power(2, np.linspace(-2, 3, 6)),    # scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
            "strfMean":0,                  # Only produce the mean activations
            "csftDensity":512,
            "csftNormalize":True
        }
        return transformList, transformOptions;
       
    def __dir__(self):
        tmpList = super(DatasetAudio, self).__dir__()
        return tmpList + ['importBatchSize', 'transformType', 'matlabCommand']
    
    def plotExampleSet(self, setData, labels, task, ids):
        fig = plt.figure(figsize=(12, 24))
        ratios = np.ones(len(ids))
        fig.subplots(nrows=len(ids),ncols=1,gridspec_kw={'width_ratios':[1], 'height_ratios':ratios})
        for ind1 in range(len(ids)):
            ax = plt.subplot(len(ids), 1, ind1 + 1)
            if (setData[ids[ind1]].ndim == 2):
                ax.imshow(np.flipud(setData[ids[ind1]]), interpolation='nearest', aspect='auto')
            else:
                tmpData = setData[ids[ind1]]
                for i in range(setData[ids[ind1]].ndim - 2):
                    tmpData = np.mean(tmpData, axis=0)
                ax.imshow(np.flipud(tmpData), interpolation='nearest', aspect='auto')
            plt.title('Label : ' + str(labels[task][ids[ind1]]))
            ax.set_adjustable('box-forced')
        fig.tight_layout()
        
    def plotRandomBatch(self, task="genre", nbExamples=5):
        setIDs = np.random.randint(0, len(self.data), nbExamples)
        self.plotExampleSet(self.data, self.metadata, task, setIDs)
        
    
                
    #    a, sr = lbload(file)

"""
###################################
# External functions for transfroms and audio imports
###################################
"""       
def makeAnalysisFiles(curBatch, options):
    # Initialize matlab command
    options['backend']= options.get('backend') or 'python'
    if (options["backend"] == 'matlab'):
        # Prepare the call to matlab command
        finalCommand = options["matlabCommand"] + ' -nodesktop -nodisplay -nojvm -r '
        # Add the transform types
        transformString = "{";
        for it in range(len(options["transformType"])):
            transformString = transformString + '\'' + options["transformType"][it] + ((it < (len(options["transformType"]) - 1)) and '\',' or '\'}')
            finalCommand = finalCommand + ' "transformType=' + transformString + '; oldRoot = \'' + options["dataDirectory"] +  '\'; newRoot = \'' + options["analysisDirectory"] + '\'';
        # Find the path of the current toolbox (w.r.t the launching path)
        curPath = os.path.realpath(__file__)
        curPath = os.path.split(curPath)[0]
        # Now handle the eventual options
        if (options["transformOptions"]) and (not (options.get("defaultOptions") == True)):
            for t, v in options["transformOptions"].items():
                finalCommand = finalCommand + '; ' + t + ' = ' + str(v)
    finalData = [None] * len(curBatch)
    finalMeta = [None] * len(curBatch)
    # Parse through the set of batches
    curAnalysisFiles = [None] * len(curBatch)
    audioList = [None] * len(curBatch)
    curIDf = 0
    # Check which files need to be computed
    for i in range(len(curBatch)):
        curFile = curBatch[i]
        curAnalysisFiles[i] = os.path.splitext(curFile.replace(du.esc(options["dataDirectory"]), options["analysisDirectory"]))[0] + '.npy'
        try:
            fIDTest = open(curAnalysisFiles[i], 'r')
        except IOError:
            fIDTest = None
        if ((fIDTest is None) or (options["forceRecompute"] == True)):
            audioList[curIDf] = curFile 
            curIDf = curIDf + 1
        else: 
            fIDTest.close()
    audioList = audioList[:curIDf]
    
    # Some of the files have not been computed yet
    if (len(audioList) > 0):
        unprocessedString = ""
        if options["verbose"]:
            print("* Computing transforms ...")
        # Matlab processing for un-analyzed files
        if (options['backend'] == 'matlab'):
            for k, v in options['transformOptions'].items():
                if v is False:
                    options['transformOptions'][k] = 0
                elif v is True:
                    options['transformOptions'][k] = 1
            for f in range(len(audioList)):
                audioList[f] =  audioList[f].replace("'","''")
                unprocessedString = unprocessedString + '\'' + audioList[f] + (f < (len(audioList) - 1) and '\',' or '\'')
                audioList[f] =  audioList[f].replace("''","'")
            if options["verbose"]:
                print(str(len(audioList)) + ' analysis files not found.')
                print("Launching Matlab...")
            curCommand = finalCommand + '; audioList = {' + unprocessedString + '}; cd ' + curPath + '/cortical-audio; processSound; exit;"'
            print(curCommand)
            fileN = os.popen(curCommand)
            output = fileN.read();
            if options["verbose"]:
                print(output)
            fileN.close()
            for f in range(len(audioList)):
                curFile = os.path.splitext(audioList[f].replace(du.esc(options["dataDirectory"]), options["analysisDirectory"]))[0]
                try:
                    curData = sp.io.loadmat(curFile + '.mat');
                    curData = curData["transforms"][options["transformType"][0]][0][0]
                    np.save(curFile + '.npy', curData);
                    os.popen('rm ' + curFile.replace(' ', '\\ ') + '.mat');
                except:
                    pass
            
        elif (options['backend'] == "python"):
            computeTransform(audioList, options["transformType"], options["dataDirectory"], options["analysisDirectory"], options)
            
    
    
    
def importAudioData(curBatch, options):
    dataPrefix = options.get('dataPrefix')
    dataDirectory = options.get('dataDirectory') or dataPrefix+'/data' or ''
    analysisDirectory = options.get('analysisDirectory') 
    if analysisDirectory is None: 
        try:
            analysisDirectory = options.get('dataPrefix')
            analysisDirectory += '/analysis'
        except TypeError:
            print('[Error] Please specify an analysis directory to import audio data')
            
    transformName= options.get('transformName')
    if issubclass(type(transformName), list):
        print('[Warning] import transform %s among transforms %s'%(transformType[0], transformType))
        transformType = transformType[0]
    
    finalData = []
    finalMeta = []
    for f in curBatch:
        if transformName == 'raw':
            finalData.append(importRawSignal(f, options))
            continue
        curAnalysisFile = re.sub(dataDirectory, analysisDirectory+'/'+transformName, f)
        curAnalysisFile = os.path.splitext(curAnalysisFile)[0] + '.npy'
        finalData.append(np.load(curAnalysisFile))
        finalMeta.append(0);
    return finalData, finalMeta


    