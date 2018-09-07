"""

    The ``Generic dataset import`` module
    ========================
 
    This file contains the basic definitions for any type of dataset.
    Importing a dataset usually relies on the same set of functions
        * Find files in a directory
        * List files from a metadata file
        * Import data from the files
        * Perform transform
        * Augment the data

    Example
    -------
    >>> from distances import timeseries as dts
    >>> dts.lcss(1, 1)
    2
 
    Currently implemented
    ---------------------
    __init__(options)
        Class constructor
    listDirectory()
        Fill the list of files from a direct recursive path of the dataDirectory attribute
    listInputFile(inputFile)
        List informations over a dataset (based on a given inputs.txt file)
    importMetadata(fileName, task, callback)
        Import metadata from filename, task and callback
    importMetadataTasks()
        Perform all metadata imports based on pre-filled tasks
    importData()
        Import all the data files directly
    importDataAsynchronous(dataIn, options)
        Import the data files asynchronously
    createSubsetsPartitions(task)
        Create subsets of data from the partitions
    constructPartition(tasks, partitionNames, partitionPercent, balancedClass, equalClass)
        Construct a random/balanced partition set for each dataset
    constructPartitionFiles(self, partitionNames, partitionFiles):
        Constructing partitions from a given set of files
    augmentDataAsynchronous(dataIn, options):
        Perform balanced and live data augmentation for a batch
    get(i)
        Access a single element in the dataset 
    createBatches(partition, batchSize, balancedTask)
        Create a set of batches balanced in terms of classes
    flattenData(self, vectorizeResult)
        Flatten the entire data matrix
    filterSet(datasetIDx, currentMetadata, curSep)
        Filter the dataset depending on the classes present
    
    Comments and issues
    -------------------
    None for the moment

    Contributors
    ------------
    Philippe Esling (esling@ircam.fr)
    
"""

import re, pdb
import numpy as np
from os import walk
from os import path
import os
# Package-specific imports
from .metadata import metadataCallbacks
from .utils import tensorDifference, tensorIntersect
import torch.utils.data

class Dataset(torch.utils.data.Dataset):    
    """ 
    
    Definition of a basic dataset object
    
    Note
    ----
    This class should be avoided, check for more specialized classes

    Attributes
    ----------
    dataDirectory : str
        Path to the current dataset
    analysisDirectory : str
        Path to the analysis (transforms if needed)
    metadataDirectory : str
        Path to the metadata
    dataPrefix : str
        Path to relative analysis of metadata
    importType : str
        Type of importing (asynchronous, direct)
    importCallback : function pointer
        Function pointer for importing data
    types : list of str
        List of all accepted filetypes
    tasks : list of str
        List of the supervised tasks for which to find metadata
    taskCallback : list of function pointers
        List of functions to import the metadata related to each task
    partitions : list of numpy arrays
        Sets of partitions for valid, test or train style of splits
    verbose : bool
        Activate verbosity to print all information
    forceUpdate : bool
        Force to recompute transforms on all files
    checkIntegrity : bool
        Check that all files are correct
    augmentationCallbacks : list of function pointers
        Set of data augmentation functions
    hash : dict
        Dictionnary linking files to their data indices
    files : list of str
        Sets of paths to the dataset files
    classes : dict of numpy array
        Dict of class names along with their indices
    metadata : dict of numpy array
        Dict of metadatas for all tasks
    labels : dict of numpy array
        Lists of labels (related to classes indices)
    data : list of numpy arrays
        Set of the data for all files
    metadataFiles : list of str
        List of all files for metadata
        
    See also
    --------
    datasetAudio, datasetMidi

    """

    def __init__(self, options):
        """ 
        Class constructor 
        
        Parameters
        ----------
        options : dict

        Returns
        -------
        A new dataset object
 
        Example
        -------
        """
        # Directories
        self.dataPrefix = options.get("dataPrefix") or ''
        self.dataDirectory = options.get("dataDirectory") or (self.dataPrefix + '/data')
        self.analysisDirectory = options.get("analysisDirectory") or (self.dataPrefix + '/analysis')
        self.metadataDirectory = options.get("metadataDirectory") or (self.dataPrefix + '/metadata')
        # Type of import (direct, asynchronous)
        self.importType = options.get("importType") or 'asynchronous'
        self.importCallback = options.get("importCallback") or {}
        # Accepted types of files
        self.types = options.get("types") or ['mp3', 'wav', 'wave', 'aif', 'aiff', 'au']
        # Tasks to import
        self.tasks = options.get("tasks") or []
        self.taskCallback = [None] * len(self.tasks)
        # Partitions in the dataset
        self.partitions = {}
        # Filters in the dataset
        # Properties of the dataset
        self.verbose = options.get("verbose") or False
        self.forceUpdate = options.get("forceUpdate") or False
        self.checkIntegrity = options.get("checkIntegrity") or False;
        # Augmentation callbacks (specific to the types)
        self.augmentationCallbacks = [];
        self.hash = {}
        self.files = []
        self.classes = {}
        self.metadata = {} 
        self.labels = []
        self.data = []
        self.metadataFiles = [None] * len(self.tasks)
        for t in range(len(self.tasks)):
            self.taskCallback[t] = (options.get("taskCallback") and options["taskCallback"][t]) or metadataCallbacks[self.tasks[t]] or metadataCallbacks["default"] or []
            self.metadataFiles[t] = (options.get("metadataFiles") and options["metadataFiles"][t]) or self.metadataDirectory + '/' + self.tasks[t] + '/metadata.txt' or self.metadataDirectory + '/metadata.txt'

    def __getitem__(self, idx):
        if type(idx) == str:
            if not idx in self.partitions.keys():
                raise IndexError('%s is not a partition of current dataset'%idx)
            return torch.utils.data.dataset.Subset(self, self.partitions[idx]) 
        else:
            item = self.data[idx]
            return item


    
    def __len__(self):
        return len(self.data)
    
    def set_active_task(self, task):
        if not task in self.tasks:
            raise Exception('task %s not in dataset tasks'%task)
        self.active_task = task
    """
    ###################################
    
    Listing functions
    
    ###################################
    """

    def listDirectory(self):
        """ 
        Fill the list of files from a direct recursive path of a folder.
        This folder is specified at object construction with the 
        dataDirectory attribute
 
        See also
        --------
        listInputFile()
        
        """
        # The final set of files
        filesList = []
        hashList = []
        # Use glob to find all recursively
        print(self.dataDirectory)
        for dirpath,_,filenames in os.walk(self.dataDirectory):
            for f in filenames:
                if f.endswith(tuple(self.types)):
                    filesList.append(os.path.abspath(os.path.join(dirpath, f)))
        hashList = {}
        curFile = 0;
        # Parse the list to have a hash
        for files in filesList:
            hashList[files] = curFile
            curFile = curFile + 1
        # Print info if verbose mode
        if (self.verbose):
            print('[Dataset][List directory] Found ' + str(curFile) + ' files.');
        # Save the lists
        self.files = filesList;
        self.hash = hashList;
    
    def listInputFile(self, inputFile):
        """ 
        List informations over a dataset (based on a given inputs.txt file).
        Will fill the files atributte of the instance
        
        Parameters
        ----------
        inputFile : str
            Path to the file containing the list of datafiles to import
        
        """
        inputFile = inputFile or self.metadataDirectory + '/inputs.txt'
        # Try to open the given file
        fileCheck = open(inputFile, "r")
        if (fileCheck is None):
            print('[Dataset][List file] Error - ' + inputFile + ' does not exists !.')
            return None
        # Create data structures
        filesList = []
        hashList = {}
        curFile = 0
        testFileID = None
        for line in fileCheck:
            if line[0] != "#":
                vals = re.search("^([^\t]+)\t?(.+)?$", line)
                audioPath = vals.group(1)
                if (audioPath is None): 
                    audioPath = line
                if (self.checkIntegrity):
                    testFileID = open(self.dataPrefix + '/' + audioPath, 'r')
                if (self.checkIntegrity) and (testFileID is None):
                    if (self.verbose): 
                        print('[Dataset][List file] Warning loading ' + inputFile + ' - File ' + self.dataPrefix + audioPath + ' does not exists ! (Removed from list)')
                else:
                    if (testFileID):
                        testFileID.close()
                    if (self.hash.get(self.dataPrefix + '/' + audioPath) is None):
                        filesList.append(self.dataPrefix + '/' + audioPath)
                        hashList[filesList[curFile]] = curFile
                        curFile = curFile + 1
        fileCheck.close()
        # Save the lists
        self.files = filesList
        self.hash = hashList

    """
    ###################################
    
    # Metadata loading functions
    
    ###################################
    """

    def importMetadata(self, fileName, task, callback):
        """  
        Import the metadata given in a file, for a specific task
        The function callback defines how the metadata should be imported
        All these callbacks should be in the importUtils file 
        
        Parameters
        ----------
        fileName : str
            Path to the file containing metadata
        task : str
            Name of the metadata task
        callback : function pointer
            Callback defining how to import metadata

        """
        # Try to open the given file
        try:
            fileCheck = open(fileName, "r")
        except:
            print('[Dataset][List file] Error - ' + fileName + ' does not exists !.')
            return None
        # Create data structures
        metaList = [None] * len(self.files)
        curFile, curHash = len(self.files), -1
        testFileID = None
        classList = {"_length":0}
        for line in fileCheck:
            line = line[:-1]
            if line[0] != "#" and len(line) > 1:
                vals = line.split('\t') #re.search("^(.+)\t(.+)$", line)
                audioPath, metaPath = vals[0], (vals[1] or "") #vals.group(1), vals.group(2)
                #print(audioPath)
                #print(metaPath)
                if (audioPath is not None):
                    fFileName = self.dataPrefix + '/' + audioPath;
                    if (self.checkIntegrity):
                        try:
                            testFileID = open(fFileName, 'r')
                        except:
                            print('[Dataset][Metadata import] Warning loading task ' + task + ' - File ' + fFileName + ' does not exists ! (Removed from list)')
                            continue
                    if (testFileID):
                        testFileID.close()
                    if (self.hash.get(fFileName) is None):
                        self.files.append(fFileName)
                        self.hash[self.files[curFile]] = curFile
                        curHash = curFile; 
                        curFile = curFile + 1
                    else:
                        curHash = self.hash[fFileName];
                    if (len(metaList) - 1 < curHash):
                        metaList.append("");
                    metaList[curHash], classList = callback(metaPath, classList, {"prefix":self.dataPrefix, "files":self.files})
                else:
                    metaList.append({})
        # Save the lists
        self.metadata[task] = np.array(metaList);
        self.classes[task] = classList;

    def importMetadataTasks(self):
        """ Perform all metadata imports based on pre-filled tasks """
        for t in range(len(self.tasks)):
            self.importMetadata(self.metadataFiles[t], self.tasks[t], self.taskCallback[t])

    """
    ###################################
    
    # Data import functions
    
    ###################################
    """
    
    def importData(self):
        """ Import all the data files directly """
        print('[Dataset][Import] Warning, undefined function in generic dataset.')
        self.data = []

    def importDataAsynchronous(self, dataIn, options):
        """ Import the data files asynchronously """
        print('[Dataset][Import (async)] Warning, undefined function in generic dataset.')
        self.data = []

    """
    ###################################
    
    # Partitioning functions
    
    ###################################
    """

    def createSubsetsPartitions(self, task):
        """ 
        Create subsets of data from the partitions.
        Based on the ``partitions`` attribute, return partitioned data and labels
        
        Parameters
        ----------
        task : str
            Name of the task to partition on
        """
        dataSubsets = {};
        # fTask = task or self.tasks[0]
        for k, v in self.partitions.iteritems():
            dataSubsets[k] = {};
            dataSubsets[k]["data"] = [None] * v.shape[0]
            dataSubsets[k]["labels"] = [None] * v.shape[0]
            for i in range(v.shape[0]):
                dataSubsets[k]["data"][i] = self.data[v[i]];
                dataSubsets[k]["labels"][i] = self.metadata[task][v[i]];
        return dataSubsets
    
    def constructPartition(self, tasks, partitionNames, partitionPercent, balancedClass=True, equalClass=False):
        """
        Construct a random/balanced partition set for each dataset
        Only takes indices with valid metadatas for every task
        now we can only balance one task
        
        Parameters
        ----------
        tasks : list of str
            Sets of task names to partition on
        partitionName : list of str
            Sets of partition names
        partitionPercent : list of float
            Sets of partition percent
        balancedClass : bool
            Mirror the overall class representation across all partitions
        equalClass : bool
            Enforce partitions with an equal number of each class
        
        """
        if (type(tasks) is str):
            tasks = [tasks]
        if (balancedClass): 
            balancedClass = tasks[0]
        # Checking if tasks exist
        for t in tasks:
            if (self.metadata[t] is None): 
                print("[Dataset] error creating partitions : " + t + " does not seem to exist")
                return None
        # making temporary index from mutal information between tasks
        mutual_ids = [];
        for i in range(self.data.shape[0]):
            b = True
            for t in tasks: 
                b = b and (self.metadata[t][i] is not None)
            if (b):
                mutual_ids.append(i)
        # Number of instances to extract
        nbInstances = len(mutual_ids)
        if (len(mutual_ids) == 0):
            if type(self.metadata[tasks[1]]) is np.ndarray:
                nbInstances = (self.metadata[tasks[0]].shape[0])
            else: 
                nbInstances = len(self.metadata[tasks[0]])
            for i in range(nbInstances):
                mutual_ids[i] = i
        partitions = {}
        runningSum = 0;
        partSizes = np.zeros(len(partitionNames))
        for p in range(len(partitionNames)):
            partitions[partitionNames[p]] = [];
            if (p != len(partitionNames)):
                partSizes[p] = np.floor(nbInstances * partitionPercent[p]);
                runningSum = runningSum + partSizes[p];
            else:
                partSizes[p] = nbInstances - runningSum;
        # Class-balanced version
        if balancedClass:
            # Perform class balancing
            curMetadata = self.metadata[balancedClass];
            curClasses = self.classes[balancedClass];
            nbClasses = curClasses["_length"];
            countclasses = np.zeros(nbClasses);
            classIDs = {};
            # Count the occurences of each class
            for idC in range(len(mutual_ids)):
                s = mutual_ids[idC]
                countclasses[curMetadata[s]] = countclasses[curMetadata[s]] + 1;
                # Record the corresponding IDs
                if (not classIDs.get(curMetadata[s])):
                    classIDs[curMetadata[s]] = [];
                classIDs[curMetadata[s]].append(s);
            if equalClass:
                minCount = np.min(countclasses) 
                for c in range(nbClasses):
                    countclasses[c] = int(minCount);
            for c in range(nbClasses):
                if (classIDs[c] is not None):
                    curIDs = np.array(classIDs[c]);
                    classNb, curNb = 0, 0;
                    shuffle = np.random.permutation(int(countclasses[c]))
                    for p in range(len(partitionNames)):
                        if equalClass:
                            classNb = np.floor(partSizes[p] / nbClasses); 
                        else:
                            classNb = np.floor(countclasses[c] * partitionPercent[p])
                        if (classNb > 0):
                            for i in range(int(curNb), int(curNb + classNb - 1)):
                                partitions[partitionNames[p]].append(curIDs[shuffle[np.min([i, shuffle.shape[0]])]])
                            curNb = curNb + classNb;
        else:
            # Shuffle order of the set
            shuffle = np.random.permutation(len(mutual_ids))
            curNb = 0
            for p in range(len(partitionNames)):
                part = shuffle[int(curNb):int(curNb+partSizes[p]-1)]
                for i in range(part.shape[0]):
                    partitions[partitionNames[p]].append(mutual_ids[part[i]])
                curNb = curNb + partSizes[p];
        for p in range(len(partitionNames)):
            self.partitions[partitionNames[p]] = np.array(partitions[partitionNames[p]]);
        return partitions;



    def constructPartitionFiles(self, partitionNames, partitionFiles):
        """ 
        Constructing partitions from a given set of files.
        Each of the partition file given should contain a list of files that
        are present in the original dataset list
        
        Parameters
        ----------
        partitionNames : list of str
            List of the names to be added to the ``partitions`` attribute
        partitionFiles : list of str
            List of files from which to import partitions
        
        """
        def findFilesIDMatch(fileN):
            fIDRaw = open(fileN, 'r');
            finalIDx = []
            if (fIDRaw is None):
                print('  * Annotation file ' + fileN + ' not found.');
                return None
            # Read the raw version
            for line in fIDRaw:
                data = line.split("\t")
                pathV, fileName = path.split(data[0])
                fileName, fileExt = path.splitext(fileName)
                for f in range(len(self.files)):
                    path2, fileName2 = path.split(self.files[f])
                    fileName2, fileExt2 = path.splitext(fileName2)
                    if (fileName == fileName2):
                        finalIDx.append(f)
                        break;
            return np.array(finalIDx);
        for p in range(len(partitionNames)):
            self.partitions[partitionNames[p]] = findFilesIDMatch(partitionFiles[p]);
        return self.partitions;

    def augmentDataAsynchronous(self, dataIn, options):
        """
        Perform balanced and live data augmentation for a batch
            * Create for each example a sub-batch (based on amount of classes)
            * Noise, outlier and warping (for axioms of robustness)
            * Eventually crop, scale (sub-sequence selections)
        Should plot the corresponding manifold densification
        """
        # First collect the options
        targetSize = options["targetSize"]
        balanced = options.get("balanced") or True
        # First collect the current number of examples
        trainData = dataIn["data"]
        trainLabels = dataIn["labels"]
        curSize = trainData.shape[0]
        # Densify the dataset
        seriesMissing = targetSize - curSize;
        print("    * Densifying set with " + str(seriesMissing) + " new series")
        if (seriesMissing > 0):
            # First prepare a balanced set
            if (balanced):
                nbClasses = np.max(trainLabels)
                countLabels = np.zeros(nbClasses)
                classIDs = {}
                # Count the occurences of each class
                for s in range(trainLabels.shape[0]):
                    countLabels[trainLabels[s]] = countLabels[trainLabels[s]] + 1
                    # Record the corresponding IDs
                    if (classIDs[trainLabels[s]] is None):
                        classIDs[trainLabels[s]] = {}
                    classIDs[trainLabels[s]].append(s)
                maxCount = np.max(countLabels)
                replicatedIDSet = np.zeros(maxCount * nbClasses);
                curID = 1;
                nbValidClasses = 1;
                for c in range(nbClasses):
                    if (classIDs[c] is not None):
                        curIDs = np.array(classIDs[c])
                        curRandSet = curIDs.index(1, np.random.rand(maxCount).mul(curIDs.shape[0]).floor().add(1))
                        replicatedIDSet[curID:curID+maxCount-1] = curRandSet;
                        nbValidClasses = nbValidClasses + 1;
                        curID = curID + maxCount;
                replicatedIDSet = replicatedIDSet[:curID-1];
                newSeriesID = replicatedIDSet.index(1, np.random.rand(seriesMissing).mul(curID - 1).floor().add(1))
            else:
                newSeriesID = np.random.rand(seriesMissing).mul(curSize).floor().add(1)
            # Use the complete size dimensions to prepare the
            dataDims = trainData.shape()
            # Replace the first dimension
            dataDims[0] = seriesMissing
            newData = np.zeros(dataDims);
            newLabels = trainLabels.index(1, newSeriesID)
            # Compute the size of a single data point
            fullSize = 1
            for f in range(1,trainData.nDimension):
                fullSize = fullSize * trainData.shape[f]
            for s in range(seriesMissing):
                tmpSeries = trainData[newSeriesID[s], :].copy();
                # Randomly draw an augmentation callback
                curDensification = np.floor(np.random.rand() * len(self.augmentationCallbacks));
                # Call it on the current data
                newData[s] = self.augmentationCallbacks[curDensification + 1](tmpSeries);
            finalSet = {};
            finalSet["data"] = np.hstack(trainData, newData)
            finalSet["labels"] = np.hstack(trainLabels, newLabels)
        else:
            finalSet = trainData;
        return finalSet;

    """
    ###################################
    #
    # Data indexing functions
    #
    ###################################
    """

    def get(self, i):
        """ Access a single element in the dataset """
        # On-the-fly import
        if (self.data[i] is None):
            self.importData([i], {})
        return self.data[i];

    def createBatches(self, partition, batchSize, balancedTask):
        """
        Create a set of batches balanced in terms of classes
        modif axel 31/04/17 : choix de la partition
        (for non balanced batches just let task to nil)
        (to fetch the whole dataset let partition to nil)
        (#TODO what if I could put myself to nil
        """
        finalIDs = {};
        if balancedTask:
            if balancedTask == True:
                balancedTask = self.task[1]
            if (self.metadata[balancedTask] is None):
                print('[Dataset] Error creating batch : ' + balancedTask + ' does not seem to exist') 
                return None
            if partition:
                partition_ids = self.partitions[partition] 
            else:
                partition_ids = range(1, len(self.metadata[balancedTask]))
            labels = np.array(self.metadata[balancedTask])[partition_ids]
            nbClasses = self.classes[balancedTask]["_length"]
            countLabels = np.zeros(nbClasses)
            classIDs = {};
            # Count the occurences of each class
            for s in range(partition_ids.shape[0]):
                countLabels[labels[s]] = countLabels[labels[s]] + 1;
                # Record the corresponding IDs
                if (classIDs.get(labels[s]) is None):
                    classIDs[labels[s]] = [];
                classIDs[labels[s]].append(partition_ids[s])
            minClassNb = np.min(countLabels)
            finalIDs = np.zeros(int(minClassNb * nbClasses));
            # Then perform a randperm of each class
            for c in range(nbClasses):
                curIDs = np.array(classIDs[c])
                curIDs = curIDs[np.random.permutation(curIDs.shape[0])]
                setPrep = (np.linspace(0, ((minClassNb - 1) * nbClasses), minClassNb) + (c - 1)).astype(int)
                finalIDs[setPrep] = curIDs[:int(minClassNb)]
            # Return class-balanced IDs split by batch sizes
            overSplit = finalIDs.shape[0] % batchSize
            finalIDs = np.split(finalIDs[:-overSplit], finalIDs[:-overSplit].shape[0] / batchSize);
        else:
            if partition:
                partition_ids = self.partitions[partition] 
            else:
                partition_ids = range(1, len(self.data))
            indices = np.random.permutation(partition_ids.shape[0])
            curIDs = partition_ids[indices]
            overSplit = curIDs.shape[0] % batchSize
            finalIDs = np.split(curIDs[:-overSplit], curIDs[:-overSplit].shape[0] / batchSize)
        # Remove the last if it is smaller
        # if finalIDs[#finalIDs]:size() < batchSize then finalIDs[#finalIDs] = nil; end
        return finalIDs;

    def flattenData(self, selector=lambda x: x):
        dataBuffer = []
        newMetadata = {}
        for k, v in self.metadata.items():
            newMetadata[k] = []
        newFiles = []
        revHash = {}
        # new hash from scratch
        newHash = dict(self.hash)
        for k, v in self.hash.items():
            newHash[k] = []
        # filter dataset
        running_sum = 0
        min_size = None
        for i in range(len(self.data)):
            # update minimum content shape
            if min_size is None:
                min_size = self.data[0].shape[1]
            else:
                if self.data[i].shape[1]<min_size:
                    min_size = self.data[i].shape[1]
            chunk_to_add = selector(self.data[i])
            if chunk_to_add.ndim == 1:
                chunk_to_add = np.reshape(chunk_to_add, (1, chunk_to_add.shape[0]))
            dataBuffer.append(chunk_to_add)
            for k, _ in newMetadata.items():
                newMetadata[k].extend([self.metadata[k][i]]*dataBuffer[i].shape[0])
            newFiles.extend([self.files[i]]*dataBuffer[i].shape[0])
            running_sum += dataBuffer[i].shape[0]
        newData = np.zeros((running_sum, min_size), dtype=self.data[0].dtype)
        running_id = 0
        for i in range(len(dataBuffer)):
            newData[running_id:(running_id+dataBuffer[i].shape[0]), :] = dataBuffer[i][:, :min_size]
            running_id+=dataBuffer[i].shape[0]
            newHash[self.files[i]].extend(range(running_id, running_id+dataBuffer[i].shape[0]))
            for idx in range(running_id, running_id+dataBuffer[i].shape[0]):
                revHash[idx] = i
        self.data = newData
        self.metadata = newMetadata
        for k,v in newMetadata.items():
            newMetadata[k] = np.array(v)
        self.files = newFiles
        self.hash = newHash
        self.revHash = revHash
            
        # flatten data
                

    """
    ###################################
    #
    # Compatibility functions
    #
    ###################################
    """
    
    def toPytorch(self):
        """ Use duck typing to transform to Pytorch """
        self.__class__ = DatasetPytorch
        return self
    
    def toTensorflow(self):
        """Tensorflow is more tricky, need actual transfer of properties """
        tupFeat = ()
        for t in dir(self):
            tupFeat = tupFeat + self.t
        tmpSet = DatasetTensorflow.createTFDataset(tupFeat)
        tmpSet.__class__ = DatasetTensorflow
        return tmpSet

    def __dir__(self):
        return ['dataDirectory', 'analysisDirectory', 'metadataDirectory',
                'dataPrefix', 'types', 'tasks', 'partitions', 'hash',
                'files', 'classes', 'metadata', 'labels', 'data']

    """
    ###################################
    #
    # Filtering functions
    #
    ###################################
    """

    def filterSet(self, datasetIDx, currentMetadata, curSep):
        testSets = {};
        # Separate set into single values
        if (curSep == 'Single'):
            for i in range(datasetIDx.shape[0]):
                testSets[i] = np.zeros(1).fill(datasetIDx[i])
        # Separate set into randomly drawn thirds
        elif (curSep == 'Random'):
            nbTrain = datasetIDx.shape[0] / 3;
            for i in range(10):
                testSets[i] = datasetIDx.index(1, np.randperm(datasetIDx.shape[0])[:nbTrain])
        # Separate on metadata
        else:
            testSets = currentMetadata[curSep + "IDSet"]
            curSetK = 1
            for i in range(len(testSets)):
                tmpSet = tensorIntersect(testSets[i], datasetIDx);
                if (tmpSet is not None):
                    testSets[curSetK] = tmpSet
                    curSetK = curSetK + 1
        # Construct corresponding test sets
        trainSets = {};
        for i in range(len(testSets)):
            trainSets[i] = tensorDifference(datasetIDx, testSets[i])
        return trainSets, testSets;
        
"""
###################################
# Pytorch dataset
###################################
"""
import torch.utils.data as tud

class DatasetPytorch(Dataset, tud.Dataset):    
    """ 
    Definition of a basic Pytorch dataset compatible with ours
    
    torch.utils.data.Dataset is an abstract class representing a dataset. 
    The custom dataset inherits Dataset and override the following methods:
        * __len__ so that len(dataset) returns the size of the dataset.
        * __getitem__ to support the indexing such that dataset[i] can be used to get iith sample

    """

    def __init__(self, options):
        super(DatasetPytorch, self).__init__(options)
        
    def __len__(self):
        """ Returns the length of the dataset """
        return max(len(self.files), len(self.data))

    def __getitem__(self, idx):
        """ Allows to index the class such as dataset[i] """
        if (self.files is None or len(self.files) == 0):
            raise Exception("Trying to get item on empty dataset")
        if (len(self.data) < idx):
            self.importData([idx])
        sample = {'data': self.data[idx]}
        if (self.tasks):
            sample['label'] = self.metadata[self.tasks[0]][idx]
        if self.transform:
            sample = self.transform(sample)
        return sample



"""
###################################
# Tensorflow dataset
###################################
"""


try:
    import tensorflow.contrib.data as tfd
    class DatasetTensorflow(Dataset, tfd.Dataset):
        """ Definition of a basic dataset
        Attributes:
            dataDirectory: 
        """

        def __init__(self, options):
            super(DatasetTensorflow, self).__init__(options)

        @staticmethod
        def createTFDataset(tupFeat):
           return tfd.Dataset.from_tensor_slices(tupFeat)
except:
    print("[WARNING] error while importing tensoprflow in %s ; please verify your install"%locals()['__name__'])
