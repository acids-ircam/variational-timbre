"""

 Import toolbox       : Import utilities
 
 This file contains lots of useful utilities for dataset import.
 
 Author               : Philippe Esling
                        <esling@ircam.fr>

"""

import numpy as np
import os
import re
# Package-specific imports
from . import audio
from .metadata import metadataCallbacks

def esc(x):
   """ Handles escape characters in string matching """
   # Define desired replacements here
   rep = {'%%':'%%%%','^%^':'%%^', '%$$':'%%$','%(':'%%(','%)':'%%)', '%.':'%%.',
          '%[':'%%[','%]':'%%]','%*':'%%*','%+':'%%+','%-':'%%-','%?':'%%?'}
   # use these three lines to do the replacement
   rep = dict((re.escape(k), v) for k, v in rep.items())
   pattern = re.compile("|".join(rep.keys()))
   return pattern.sub(lambda m: rep[re.escape(m.group(0))], x)

def tableToTensor(table):
    """  Transform a table of tensors to a single tensor """
    tensorSize = table[0].shape
    tensorSizeTable = [-1]
    for i in range(tensorSize.shape[0]):
        tensorSizeTable[i + 1] = tensorSize[i]
    merge = nn.Sequential().add(nn.JoinTable(1)).add(nn.View(unpack(tensorSizeTable)));
    return merge.forward(table);

def windowData(inputData, windowDim=0, wSize=16, wStep=1):
    """
    Create a set of windows over a given temporal transforms
    The returned data table does not allocate memory to enhance performance !
    """
    # Just skip empty data
    if ((inputData is None) or (inputData[1] is None)): 
        return None
    tDim = windowDim
    size = wSize
    step = wStep
    # Compute the number of windows
    rep = np.ceil((inputData.shape[tDim] - size + 1) / step)
    sz = np.zeros(inputData.nDim + 1)
    currentOutput = np.zeros((rep, inputData.shape[0], size))
    # Perform a set of narrows
    for i in range(rep):
        currentOutput[i] = inputData.narrow(tDim, ((i - 1) * step + 1), size)
    return currentOutput;

def mkdir(path):
    """ Create a directory """
    assert(os.popen('mkdir -p %s' % path), 'could not create directory')

def tensorDifference(ref, rem):
    """ Returns the difference between two tensors """
    finalTensor = np.zeros(ref.shape[0])
    i = 1
    for j in range(ref.shape[0]):
        found = 0
        for k in range(rem.shape[0]):
            if (rem[k] == ref[j]):
                found = 1
        if (found == 0):
            finalTensor[i] = ref[j]
            i = i + 1
    return finalTensor[:i]

def tensorIntersect(ref, rem):
    """ Returns the intersection between two tensors """
    finalTensor = np.zeros(ref.shape[0])
    i = 1;
    for j in range(ref.shape[0]):
        found = 0;
        for k in range(rem.shape[0]):
            if (rem[k] == ref[j]):
                found = 1
        if (found == 1):
            finalTensor[i] = ref[j]
            i = i + 1
    if (i == 1):
        return None
    return finalTensor[:i-1]

def tableDifference(ref, rem):
    """ Returns the difference between two tables """
    finalTable = []
    for j in range(len(ref)):
        found = 0;
        for k in range(len(rem)): 
            if (rem[k] == ref[j]):
                found = 1
        if (found == 0): 
            finalTable.append(ref[j])
    return finalTable;


"""
###################################
# A real resampling function for time series
###################################
function tensorResampling(data, destSize, type)
  # Set the type of kernel
  local type = type or 'gaussian'
  # Check properties of input data
  if data:dim() == 1 then
    data:resize(1, data:dim(1));
  end
  # Original size of input
  local inSize = data:size(2);
  # Construct a temporal convolution object
  local interpolator = nn.TemporalConvolution(inSize, destSize, 1, 1);
  # Zero-out the whole weights
  interpolator.weight:zeros(destSize, inSize);
  # Lay down a set of kernels
  for i = 1,destSize do
    if type == 'gaussian' then
      interpolator.weight[i] = image.gaussian1D(inSize, (1 / inSize), 1, true, i / destSize);
    else
      local kernSize = math.abs(inSize - destSize) * 3;
      # No handling of boundaries right now
      for j = math.max({i-kernSize, 1}),math.min({i+kernSize,destSize}) do
        # Current position in kernel
        local relIdx = (j - i) / kernSize;
        if type == 'bilinear' then
          interpolator.weight[i][j] = 1 - math.abs(relIdx);
        elseif type == 'hermite' then
          interpolator.weight[i][j] = (2 * (math.abs(relIdx) ^ 3)) - (3 * (math.abs(relIdx) ^ 2)) + 1;
        elseif type == 'lanczos' then
          interpolator.weight[i][j] = (2 * (math.abs(relIdx) ^ 3)) - (3 * (math.abs(relIdx) ^ 2)) + 1;
        end
      end
    end
  end
  # print(interpolator.weight);
  return interpolator:forward(data);
end

###################################
# Return the table of unique occurences (and hash values)
###################################
function tableToLabels(table)
  local hash = {}
  local res = {}
  local labels = {};
  for k,v in pairs(table) do
    if (not hash[v]) then
      res[#res+1] = v # you could print here instead of saving to result table if you wanted
      hash[v] = #res;
    end
    labels[k] = hash[v];
  end
  res[#res + 1] = '(none)';
  return labels, res;
end

###################################
# Retrieve unique values and index set from a table
###################################
function uniqueTable(table)
  local hash = {}
  local res = {}
  for k,v in pairs(table) do
    if (not hash[v]) then
      res[#res+1] = v
      hash[v] = {};
      hash[v][1] = k
    else
      hash[v][#hash[v]+1] = k;
    end
  end
  finalRes = {}; finalIDx = {}; i = 1;
  for k,v in pairs(hash) do finalRes[i] = k; finalIDx[i] = torch.Tensor(hash[k]):long(); i = i + 1; end
  return finalRes, finalIDx;
end

###################################
# Retrieve unique values of a tensor
###################################
function uniqueTensor(table)
  local hash = {}
  local res = {}
  for i = 1,table:size(1) do
    if (not hash[table[i]]) then
      res[#res+1] = table[i] # you could print here instead of saving to result table if you wanted
      hash[table[i]] = true
    end
  end
  return res;
end

###################################
# Perform a deep clone of a given object (table with sub-fields as tensors)
###################################
function deepcopy(object)  # http://lua-users.org/wiki/CopyTable
  local lookup_table = {}
  local function _copy(object)
    if type(object) ~= "table" then
      return object
    elseif lookup_table[object] then
      return lookup_table[object]
    end
    local new_table = {}
    lookup_table[object] = new_table
    for index, value in pairs(object) do
      new_table[_copy(index)] = _copy(value)
    end
    return setmetatable(new_table, getmetatable(object))
  end
  return _copy(object)
end

function round(x) return math.floor(x + 0.5) end

###################################
# Create a set of classes labels from a vector
###################################
function createClasses(valuesVector, nbClasses)
  # Create a linear split
  local yVals = torch.linspace(valuesVector:min(), valuesVector:max(), nbClasses + 1)
  local classResult = torch.ones(valuesVector:size(1)):mul(nbClasses);
  local classCounts = torch.zeros(nbClasses);
  # Find corresponding classes
  for i = 1,valuesVector:size(1) do
    for j = 1,yVals:size(1) do
      if (valuesVector[i] < yVals[j]) then
        classResult[i] = j - 1;
        classCounts[j - 1] = classCounts[j - 1] + 1;
        break;
      end
    end
  end
  validClasses = {}; k = 1;
  # Now prune the class (remove empty classes)
  for j = 1,classCounts:size(1) do
    if (classCounts[j] > 0) then validClasses[j] = k; k = k + 1; end
  end
  # Replace classes after pruning
  for i = 1,classResult:size(1) do classResult[i] = validClasses[classResult[i]]; end
  return classResult;
end

###################################
# Perform the MD5 checksum of an object
###################################
function md5(obj)
   local str = torch.serialize(obj)
   return md5.sumhexa(str)
end

# Function to recursively list files of given type
function listFiles(curDir)
  local filesList = {};
  local function listRecurse(baseDir)
    for entity in lfs.dir(baseDir) do
      if entity ~= "." and entity ~= ".." and string.sub(entity,1,1) ~= "." then
        local fullPath = baseDir + '/' .. entity
        # Check the type of the entry
        local mode=lfs.attributes(fullPath,"mode")
        if mode ~= "directory" then
          local insertID = 1;
          for i = 1,#filesList do
            if entity < filesList[i].name then break; end
            insertID = insertID + 1;
          end
          table.insert(filesList, insertID, {});
          filesList[insertID].path = fullPath;
          filesList[insertID].name = entity;
        else
          listRecurse(fullPath);
        end
      end
    end
    return filesList;
  end
  return listRecurse(curDir);
end

function getKeys(table)
  local resTable, curID = {}, 1; 
  for k, v in pairs(table) do
    resTable[curID] = k;
    curID = curID + 1;
  end
  return resTable;
end

"""

# Function to recursively list directories of given type
def listDirectories(baseDir):
    filesList = []
    insertID = 0
    for name in os.listdir(baseDir):
        path = os.path.join(baseDir, name)
        if os.path.isdir(path):
            filesList.append({})
            filesList[insertID]["path"] = path;
            filesList[insertID]["name"] = name;
            insertID = insertID + 1
    return filesList;

def getAudioFileProperty(fileName):
    fileProperties = {};
    fileProperties["channels"] = 0;
    fileProperties["rate"] = 0;
    fileProperties["size"] = 0;
    fileProperties["type"] = "";
    fileProperties["duration"] = 0;
    # Get duration of file
    pfile = os.popen('soxi "' + fileName + '"');
    for properties in pfile:
        typeV = properties[:-1].split(": ")
        if (len(typeV) > 1):
            if (typeV[0][:8] == 'Channels'):
                fileProperties["channels"] = float(typeV[1])
            if (typeV[0][:11] == 'Sample Rate'):
                fileProperties["rate"] = float(typeV[1])
            if (typeV[0][:9] == 'File Size'):
                fileProperties["size"] = float(typeV[1][:-1])
            if (typeV[0][:15] == 'Sample Encoding'):
                fileProperties["type"] = typeV[1]
    pfile.close()
    pfile = os.popen('soxi -D "' + fileName + '"')
    for properties in pfile:
        fileProperties["duration"] = float(properties)
    pfile.close()
    return fileProperties

def exportMetadataProperties(fID, task, metadata, classes):
    fID.write('#-\n' + task + '\n#-\n')
    if (metadata is None):
        return None
    if (task == 'onset'):
        print("Onset check not implemented.")
    elif (task == 'drum'):
        fID.write('Number of annotated \t : ' + str(len(metadata)) + '\n')
        fID.write('Number of classes \t : ' + str(classes["_length"]) + '\n')
        fID.write('Instance values :\n')
        tmpClasses = np.zeros(classes["_length"])
        curID = 0
        for k, v in classes.items():
            if (k != "_length"):
                nbEx = 0
                for f in range(len(metadata)):
                    if (metadata[f][0]):
                        for g in metadata[f][0]["labels"]:
                            if (g == v): 
                                nbEx = nbEx + 1
                tmpClasses[curID] = nbEx
                curID = curID + 1
        if (len(tmpClasses) == 0):
            fID.write('**\n**\n**\n WARNING EMPTY CLASSES **\n**\n**\n**\n')
            return
        fID.write('Min \t : ' + str(np.min(tmpClasses)) + '\n')
        fID.write('Max \t : ' + str(np.max(tmpClasses)) + '\n')
        fID.write('Mean \t : ' + str(np.mean(tmpClasses)) + '\n')
        fID.write('Var \t : ' + str(np.std(tmpClasses)) + '\n')
        tmpAnnote, fullTimes = np.zeros(len(metadata)), np.array([])
        for f in range(len(metadata)):
            if (metadata[f][0]):
                tmpAnnote[f] = metadata[f][0]["time"].shape[0]
                fullTimes = np.concatenate((fullTimes, metadata[f][0]["time"]))
        fID.write('Annotation lengths :\n')
        fID.write('Min \t : ' + str(np.min(tmpAnnote)) + '\n')
        fID.write('Max \t : ' + str(np.max(tmpAnnote)) + '\n')
        fID.write('Mean \t : ' + str(np.mean(tmpAnnote)) + '\n')
    elif (task == 'tempo'):
        fID.write('Number of annotated \t : ' + str(len(metadata)) + '\n')
        tmpTempo = np.array(metadata)
        fID.write('Tempo values :\n')
        fID.write('Min \t : ' + str(np.min(tmpTempo)) + '\n')
        fID.write('Max \t : ' + str(np.max(tmpTempo)) + '\n')
        fID.write('Mean \t : ' + str(np.mean(tmpTempo)) + '\n')
        fID.write('Var \t : ' + str(np.std(tmpTempo)) + '\n')
    elif (task == 'cover'):
        curID, coverTable = 0, [None] * len(metadata)
        for v in metadata:
            coverTable[curID] = len(v)
            curID = curID + 1
        fID.write('Number of annotated \t : ' + str(len(coverTable)) + '\n')
        tmpCover = np.array(coverTable)
        fID.write('Number not found (!) \t : ' + str(np.sum((tmpCover == 0) * 1)) + '\n')
        tmpCover = tmpCover[tmpCover != 0]
        fID.write('Cover properties :\n')
        fID.write('Number found \t : ' + str(tmpCover.shape[0]) + '\n')
        fID.write('Min \t : ' + str(np.min(tmpCover)) + '\n')
        fID.write('Max \t : ' + str(np.max(tmpCover)) + '\n')
        fID.write('Mean \t : ' + str(np.mean(tmpCover)) + '\n')
    elif (task == 'melody'):
        fID.write('Number of annotated \t : ' + str(len(metadata)) + '\n');
        tmpAnnote, fullTimes, fullLabels = np.zeros(len(metadata)), np.array([]), np.array([]);
        for f in range(len(metadata)):
            if (metadata[f]):
                if (len(metadata[f][0]["time"]) > 0):
                    tmpAnnote[f] = metadata[f][0]["time"].shape[0]
                    fullTimes = np.concatenate((fullTimes, metadata[f][0]["time"]))
                    fullLabels = np.concatenate((fullLabels, metadata[f][0]["labels"]))
                else:
                    fID.write('Warning : Empty metadata for ' + str(f) + '\n')
        fID.write('Annotation lengths :\n')
        fID.write('Min \t : ' + str(np.min(tmpAnnote)) + '\n')
        fID.write('Max \t : ' + str(np.max(tmpAnnote)) + '\n')
        fID.write('Mean \t : ' + str(np.mean(tmpAnnote)) + '\n')
        fID.write('Pitch values :\n')
        if (len(fullLabels) > 0):
            fID.write('Min \t : ' + str(np.min(fullLabels)) + '\n')
            fID.write('Max \t : ' + str(np.max(fullLabels)) + '\n')
            fID.write('Mean \t : ' + str(np.mean(fullLabels)) + '\n')
        else:
            fID.write('**\n**\n**\n WARNING EMPTY LABELS **\n**\n**\n**\n')
    else:
        fID.write('Number of annotated \t : ' + str(len(metadata)) + '\n')
        if (classes) and (classes["_length"] > 0):
            fID.write('Number of classes \t : ' + str(classes["_length"]) + '\n')
            fID.write('Instance values :\n')
            tmpClasses = np.zeros(classes["_length"])
            curID = 0
            for k, v in classes.items():
                if (k != "_length"):
                    nbEx = 0
                    for f in range(len(metadata)):
                        if (metadata[f] == v):
                            nbEx = nbEx + 1
                    tmpClasses[curID] = nbEx
                    curID = curID + 1
            fID.write('Min \t : ' + str(np.min(tmpClasses)) + '\n')
            fID.write('Max \t : ' + str(np.max(tmpClasses)) + '\n')
            fID.write('Mean \t : ' + str(np.mean(tmpClasses)) + '\n')
            fID.write('Var \t : ' + str(np.std(tmpClasses)) + '\n')

def testDatasetCollection(path):
    fIDm = open(path + '/datasets-metadata.txt', 'w')
    fIDt = open(path + '/datasets-tasks.txt', 'w')
    fIDt.write("%16s\t" % 'Datasets')
    taskTable, taskID = {}, 0
    print(metadataCallbacks)
    for k, v in (metadataCallbacks.items()):
        if (k != 'default'):
            taskTable[taskID] = k
            taskID = taskID + 1
            tmpKey = k
        if (len(k) > 7):
            tmpKey = k[:7]
            fIDt.write('%s\t' % tmpKey);
    fIDt.write('\n');
    # First list all datasets
    datasetsList = listDirectories(path)
    print('Found datasets :');
    for d in range(len(datasetsList)):
        print('  * ' + datasetsList[d]["name"])
        # Now check which task we found
        taskList = listDirectories(datasetsList[d]["path"] + '/metadata/')
        fIDm.write('***\n***\n' + datasetsList[d]["name"] + '\n***\n***\n\n')
        fIDt.write('%16s\t' % datasetsList[d]["name"])
        # Parse all tasks
        finalTasks = {}
        curTask = 0;
        print('    - Parsing tasks folders.')
        for t in range(len(taskList)):
            if (taskList[t]["name"] != 'raw'):
                print('      o ' + taskList[t]["name"])
                finalTasks[curTask] = taskList[t]["name"]
                curTask = curTask + 1
        wroteTask = np.zeros(len(finalTasks))
        for k in range(len(taskTable)):
            foundID = 0
            for k2 in range(len(finalTasks)):
                if (finalTasks[k2] == taskTable[k]):
                    foundID = 1
                    wroteTask[k2] = 1
                    break
            if (foundID == 1):
                fIDt.write('1')
            else:
                fIDt.write('0')
            fIDt.write('\t')
        fIDt.write('\n')
        for k in range(wroteTask.shape[0]):
            if (wroteTask[k] == 0):
                fIDt.write('ERROR - unfound task : ' + finalTasks[k] + '\n')
        print('    - Importing metadatas.')
        # Create a dataset from the task
        startPath, fileN = os.path.split(datasetsList[d]["path"])
        audioOptions = {"dataDirectory":datasetsList[d]["path"], "dataPrefix":startPath, "tasks":finalTasks };
        audioSet = audio.DatasetAudio(audioOptions)
        audioSet.importMetadataTasks()
        print('    - Storing file properties.')
        fIDm.write('#-\n' + 'File properties' + '\n#-\n')
        # Create a set of overall properties
        setRates, setTypes, setChannels, setDurations = {}, {}, {}, np.zeros(len(audioSet.files))
        # Now check properties of the files 
        for f in range(len(audioSet.files)):
            fileProp = getAudioFileProperty(audioSet.files[f]);
            if (setRates.get(fileProp["rate"])):
                setRates[fileProp["rate"]] = setRates[fileProp["rate"]] + 1
            else: 
                setRates[fileProp["rate"]] = 1
            if (setTypes.get(fileProp["type"])):
                setTypes[fileProp["type"]] = setTypes[fileProp["type"]] + 1
            else:
                setTypes[fileProp["type"]] = 1
            if (setChannels.get(fileProp["channels"])): 
                setChannels[fileProp["channels"]] = setChannels[fileProp["channels"]] + 1; 
            else: 
                setChannels[fileProp["channels"]] = 1;
            setDurations[f] = fileProp["duration"]
        fIDm.write(' * Rates\n');
        for k, v in setRates.items():
            fIDm.write('    - ' + str(k) + ' \t : ' + str(v) + '\n')
        fIDm.write(' * Types\n')
        for k, v in setTypes.items():
            fIDm.write('    - ' + str(k) + ' \t : ' + str(v) + '\n')
        fIDm.write(' * Channels\n')
        for k, v in setChannels.items():
            fIDm.write('    - ' + str(k) + ' \t : ' + str(v) + '\n')
        fIDm.write(' * Durations\n');
        fIDm.write('    - Minimum \t : ' + str(np.min(setDurations)) + '\n');
        fIDm.write('    - Maximum \t : ' + str(np.max(setDurations)) + '\n');
        fIDm.write('    - Mean \t : ' + str(np.mean(setDurations)) + '\n');
        fIDm.write('    - Variance \t : ' + str(np.std(setDurations)) + '\n');
        fIDm.write('\n');
        print('    - Metadata properties.');
        for t in range(len(finalTasks)):
            exportMetadataProperties(fIDm, finalTasks[t], audioSet.metadata[finalTasks[t]], audioSet.classes[finalTasks[t]]);
            fIDm.write('\n')
        print('    - Metadata verification.');
        fIDm.write('#-\n' + 'Metadata check (for melody,key,chord,drum and structure)' + '\n#-\n');
        for t in range(len(finalTasks)):
            # Here we will check that the annotations do not exceed the file duration !
            if (finalTasks[t] == 'melody') or (finalTasks[t] == 'keys') or (finalTasks[t] == 'chord') or (finalTasks[t] == 'drum') or (finalTasks[t] == 'structure'):
                curMeta = audioSet.metadata[finalTasks[t]]
                fIDm.write('Task ' + finalTasks[t] + '\n') 
                for f in range(len(audioSet.files)):
                    if (finalTasks[t] == 'structure') or (finalTasks[t] == 'keys') or (finalTasks[t] == 'chord') or (finalTasks[t] == 'harmony'):
                        if (not curMeta[f]) or (not curMeta[f][0]) or (len(curMeta[f][0]["timeEnd"]) == 0):
                            fIDm.write('Error : File ' + audioSet.files[f] + ' - Does not have metadata !\n')
                        elif (np.max(curMeta[f][0]["timeEnd"]) > setDurations[f]):
                            fIDm.write('Error : File ' + audioSet.files[f] + '\n')
                            fIDm.write('Duration \t : ' + str(setDurations[f]) + '\n')
                            fIDm.write('Max annote \t : ' + str(np.max(curMeta[f][0]["timeEnd"])) + '\n')
                    else:
                        if (not curMeta[f]) or (not curMeta[f][0]) or (len(curMeta[f][0]["time"]) == 0):
                            fIDm.write('Error : File ' + audioSet.files[f] + ' - Does not have metadata !\n')
                        elif (np.max(curMeta[f][0]["time"]) > setDurations[f]):
                            fIDm.write('Error : File ' + audioSet.files[f] + '\n')
                            fIDm.write('Duration \t : ' + str(setDurations[f]) + '\n')
                            fIDm.write('Max annote \t : ' + str(np.max(curMeta[f][0]["time"])) + '\n')
        fIDm.write('\n')
    fIDm.close()
    fIDt.close()