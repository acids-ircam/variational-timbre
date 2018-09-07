"""

 Import toolbox       : Metadata import utilities
 
 This file contains the generic callbacks for metadata import.
 
 Author               : Philippe Esling
                        <esling@ircam.fr>

"""

import re
import numpy as np

"""
###################################
# Generic metadata callbacks
###################################
"""

###################################
# A list of pre-defined callbacks (mostly for audio tasks)
###################################

def importRawNumber(metadata, classes, options):
    if (metadata is None):
        return None
    return float(metadata), None

def importSeries(metadata, classes, options):
    if (metadata is None):
        return None
    curID = 0
    result = {};
    while curID <= len(metadata):
        next_space = metadata.sub(curID+1).find("\n")
        line = metadata.sub(curID, curID + next_space - 1)
        # map(int, re.findall(r'\d+', line))
        findVals = re.search("^(.+)%s+(.+)$", line)
        time, values = findVals.group(1), findVals.group(2)
        if (float(time) is None) and (float(values) is None):
            result.append([float(time), map(float, values)])
        curID = curID + next_space + 1
    finalTensor = np.zeros(len(result) * 2)
    for i, elt in ipairs(result):
        finalTensor[(i - 1) * 2 + 1] = elt[0]
        finalTensor[i * 2] = elt[1]
    return finalTensor, None

def importKey(metadata, classes, options):    
    if (metadata is None):
        return None
    values = re.search("^([a-gA-G][%#b]?):([^%:]+):*(.+)$", metadata)
    key, color, extensions = values.group(1), values.group(2), values.group(3)
    if key:
        if (classes[metadata] is None):
            classes["_length"] = classes["_length"] + 1
            classes[metadata] = classes["_length"]
            return classes_, classes
        else:
            return classes[metadata], classes
    else:
        return None

def importTrackList(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = [None] * len(options["files"])
    finalID = 0
    for strFile in re.findall('([^,]+)', metadata):
        foundID = 0;
        # First look for the file
        for f in range(len(options["files"])):
            if (options["files"][f] == options["prefix"] + '/' + strFile):
                foundID = f; 
                break;
        finalVals[finalID] = foundID;
        finalID = finalID + 1;
    finalVals = finalVals[:finalID]
    return finalVals, classes;

def importLabelPairsList(metadata, classes, options):
    if (metadata is None):
        return None
    finalLabels = []
    finalVals = []
    finalID = 0
    labelFlag = 1
    for strFile in re.findall('([^,]+)', metadata):
        if (labelFlag == 1):
            finalLabels[finalID] = strFile;
            if not classes[strFile]:
                classes["_length"] = classes["_length"] + 1
                classes[strFile] = classes["_length"]
            finalLabels[finalID] = classes[strFile];
        else:
            finalVals[finalID] = float(strFile);
            finalID = finalID + 1;
        labelFlag = 1 - labelFlag;
    return {'tags':finalLabels, 'values':finalVals}, classes;

def importNumberList(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = []
    finalID = 0
    for strFile in re.findall('([^,]+)', metadata):
        finalVals[finalID] = float(strFile)
        finalID = finalID + 1
    return finalVals, classes

def importTimedLabelsFile(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = []
    finalID = 0
    for strFile in re.findall('([^,]+)', metadata):
        curVals, curLabels, curID = [], [], 0;
        fID = open(options["prefix"] + '/' + strFile, 'r');
        if (fID):
            for metaLines in fID:
                vals = metaLines.split('\t') # re.search('^(.+)\t(.+)$',
                time, val = vals[0], vals[1] #vals.group(1), vals.group(2)
                curVals.append(float(time))
                if (not classes.get(val)):
                    classes["_length"] = classes["_length"] + 1
                    classes[val] = classes["_length"]
                    idx = classes["_length"]
                else:
                    idx = classes[val]
                curLabels.append(idx)
                curID = curID + 1;
            fID.close();
            finalVals.append({});
            finalVals[finalID]["time"] = np.array(curVals);
            finalVals[finalID]["labels"] = curLabels;
            finalID = finalID + 1;
    return finalVals, classes;

def importTimedNumbersFile(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = []
    finalID = 0
    for strFile in re.findall('([^,]+)', metadata):
        curVals, curLabels, curID = [], [], 0;
        fID = open(options["prefix"] + '/' + strFile, 'r');
        if (fID):
            for metaLines in fID:
                vals = metaLines.split('\t') #re.search('^(.+)\t(.+)$', metaLines)
                time, val = vals[0], vals[1] #vals.group(1), vals.group(2)
                curVals.append(float(time))
                curLabels.append(float(val))
                curID = curID + 1;
            fID.close();
            finalVals.append({});
            finalVals[finalID]["time"] = np.array(curVals);
            finalVals[finalID]["labels"] = np.array(curLabels);
            finalID = finalID + 1;
    return finalVals, classes;

def importTimedSegmentFile(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = []
    finalID = 0
    for strFile in re.findall('([^,]+)', metadata):
        curValsS, curValsE, curLabels, curID = [], [], [], 0;
        try:
            fID = open(options["prefix"] + '/' + strFile, 'r');
        except:
            continue
        if (fID):
            for metaLines in fID:
                metaLines = metaLines[:-1]
                vals = metaLines.split(',') #re.search('^([^,]+),([^,]+),(.+)$', metaLines)
                if (len(vals) > 2):
                    timeS, timeE, val = vals[0], vals[1], vals[2] #vals.group(1), vals.group(2), vals.group(3)
                    curValsS.append(float(timeS));
                    curValsE.append(float(timeE));
                    curLabels.append(val);
                    curID = curID + 1;
            fID.close();
            finalVals.append({});
            finalVals[finalID]["timeStart"] = np.array(curValsS);
            finalVals[finalID]["timeEnd"] = np.array(curValsE);
            finalVals[finalID]["labels"] = curLabels;
            finalID = finalID + 1;
    return finalVals, classes;

def importTimedSeriesFile(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = [];
    finalID = 0;
    for strFile in re.findall('([^,]+)', metadata):
        curVals, curID = [], 0;
        fID = open(options["prefix"] + '/' + strFile, 'r');
        if (fID):
            for metaLines in fID:
                metaLines = metaLines[:-1]
                vals = metaLines.split(',');#re.search('^([^,]+),([^,]+)', metaLines);
                timeE = vals[0]
                if (len(timeE) > 0):
                    curVals.append(float(timeE)); 
                    curID = curID + 1;         
            fID.close();
            finalVals.append(np.array(curVals));
            finalID = finalID + 1;
    return finalVals, classes;

def importStringFile(metadata, classes, options):
    if (metadata is None):
        return None
    finalVals = [];
    finalID = 0;
    for strFile in re.findall('([^,]+)', metadata):
        curVals, curID = [], 0;
        fID = open(options["prefix"] + '/' + strFile, 'r');
        if (fID):
            for metaLines in fID.lines():
                val = metaLines;
                if (not classes[val]): 
                    classes["_length"] = classes["_length"] + 1
                    classes[val] = classes["_length"]
                    idx = classes["_length"]
                else:
                    idx = classes[val]
                curVals[curID] = idx; 
                curID = curID + 1;
            fID.close();
            finalVals[finalID] = np.array(curVals);
            finalID = finalID + 1;
    return finalVals, classes;

def importRawLabel(metadata, classes, options):
    if (metadata is None):
        return None
    idx = 0;
    if (classes.get(metadata) is None):
        idx = classes["_length"]
        classes[metadata] = classes["_length"]
        classes["_length"] = classes["_length"] + 1
    else:
        idx = classes[metadata]
    return idx, classes

def importRawMultiLabel(metadata, classes, options):
    if (metadata is None):
        return None
    labels = []
    for label in re.findall('([^,]+)', metadata):
        labels.append(label)
    idxs = [];
    for label in (labels):
        if (not classes.get(label)):
            classes[label] = classes["_length"]
            idxs.append(classes["_length"])
            classes["_length"] = classes["_length"] + 1;
        else:
            idxs.append(classes[label])
    return idxs, classes

"""

##############################-
# Import metadata specific for SALAMI
##############################-
function importStructure(baseFile, fftPoints, fftWins)
  # Function to import timed data
  local function createTimedLabels(labels, fftTimes, fftPoints, noneValue)
    local finalTensor = torch.Tensor(fftPoints):fill(noneValue);
    curValue = noneValue;
    for i = 1,(fftTimes:size(1) - 1) do
      if (not labels[i]) then curValue = noneValue; else curValue = labels[i]; end
      for t = fftTimes[i],(fftTimes[i + 1] - 1) do finalTensor[t] = curValue; end
    end
    return finalTensor;
  end
  # Find the base directory
  baseMeta, fileExt = string.gsub(baseFile, 'audio', 'metadata'):match("^(.+)%.(.+)$");
  # Check if directory exists
  if (lfs.attributes(baseMeta, "mode") ~= "directory") then
    print('  * ' .. baseMeta .. ' not found.');
    return nil;
  end
  metaStruct = {};
  # Check for two annotators
  for i = 1,2 do
    fIDRaw = io.open(baseMeta .. '/textfile' .. i .. '.txt', 'r');
    if (fIDRaw == nil) then
      print('  * Annotation n.' .. i .. ' for ' .. baseMeta .. ' not found.');
      if (i == 1) then return nil end;
      return metaStruct; 
    end
    metaStruct[i] = {};
    metaStruct[i].times = {};
    metaStruct[i].instruments = {};
    metaStruct[i].functions = {};
    metaStruct[i].lower = {};
    metaStruct[i].upper = {};
    curEvent = 1;
    curInstrument, curFunc, curLow, curUp = nil, nil, nil, nil;
    # Read the raw version
    for line in fIDRaw:lines('*l') do
      local time, vals = line:match("^([%d%.]+)%s*(.*)$");
      metaStruct[i].times[curEvent] = tonumber(time);
      metaStruct[i].instruments[curEvent] = curInstrument;
      metaStruct[i].functions[curEvent] = curFunc;
      metaStruct[i].lower[curEvent] = curLow;
      metaStruct[i].upper[curEvent] = curUp; 
      for val in string.gmatch(vals, "[^%,]+") do
        val = string.gsub(string.gsub(val, ' ', ''), ',', '');
        tIns = val:match("[%(|%)]");
        tFun = val:match("^([A-Z][a-z]+)");
        tUp = val:match("^([A-Z]%'?)$");
        tLow = val:match("^([a-z]%'?)$");
        if tIns then 
          if (val:match('%(')) then
            local curVal = string.gsub(string.gsub(val, '%)', ''), '%(', '');
            metaStruct[i].instruments[curEvent] = curVal;
            curInstrument = curVal;
          end
          if (val:match('%)')) then curInstrument = nil; end
        end
        if tFun then metaStruct[i].functions[curEvent] = val; curFunc = val; end
        if tLow then metaStruct[i].lower[curEvent] = val; curLow = val; end
        if tUp then metaStruct[i].upper[curEvent] = val; curUp = val; end
      end
      curEvent = curEvent + 1;
    end
    fIDRaw:close();
    # Turn the metadata into unique structures and labels
    metaStruct[i].instruments, metaStruct[i].instrumentsLabels = tableToLabels(metaStruct[i].instruments);
    metaStruct[i].functions, metaStruct[i].functionsLabels = tableToLabels(metaStruct[i].functions);
    metaStruct[i].lower, metaStruct[i].lowerLabels = tableToLabels(metaStruct[i].lower);
    metaStruct[i].upper, metaStruct[i].upperLabels = tableToLabels(metaStruct[i].upper);
    local fftPerSec = (fftPoints - 1) / metaStruct[i].times[#metaStruct[i].times];
    metaStruct[i].fftTimes = {};
    # Put time relative to FFT windows
    for p = 1,#metaStruct[i].times do
      metaStruct[i].fftTimes[p] = math.floor(metaStruct[i].times[p] * fftPerSec) + 1;
    end
    metaStruct[i].fftTimes = torch.Tensor(metaStruct[i].fftTimes);
    metaStruct[i].instruments = createTimedLabels(metaStruct[i].instruments, metaStruct[i].fftTimes, fftPoints, #metaStruct[i].instrumentsLabels);
    metaStruct[i].functions = createTimedLabels(metaStruct[i].functions, metaStruct[i].fftTimes, fftPoints, #metaStruct[i].functionsLabels);
    metaStruct[i].lower = createTimedLabels(metaStruct[i].lower, metaStruct[i].fftTimes, fftPoints, #metaStruct[i].lowerLabels);
    metaStruct[i].upper = createTimedLabels(metaStruct[i].upper, metaStruct[i].fftTimes, fftPoints, #metaStruct[i].upperLabels);
  end
  return metaStruct;
end

##############################-
# Import metadata for similarity task
##############################-
function importSimilarity(dataRoot, trainPercent)
  # Sound types
  local soundTypes = {'wav', 'wave', 'aif', 'aiff', 'mp3', 'au'} or sndTypes
  # List of files to process
  local fullDataset = {};
  # List of genre labels and genre hashmap
  local genresHash, genresList = {}, {};
  # List of artists and artist hashmap
  local artistsHash, artistsList = {}, {};
  # List of similarities and similarity
  local filesList, filesHash, filesSimilar, filesScores, hasSimilar, similarVector = {}, {}, {}, {}, {}, {};
  # Location of metadatas
  genreFile = dataRoot .. '/metadata/genre/metadata.txt';
  artistFile = dataRoot .. '/metadata/artist-identification/metadata.txt';
  similarityFile = dataRoot .. '/metadata/song-similarity/metadata.txt';
  #### GENRE IMPORT ####-
  local curFile = 0;
  fIDGenre = io.open(genreFile, 'r');
  if (fIDGenre == nil) then print('  * Annotation file ' .. genreFile .. ' not found.'); return nil; end
  # Read the raw version
  for line in fIDGenre:lines('*l') do
    data = line:split("\t");
    curFile = curFile + 1;
    # These metadatas are full path indexed
    local path, fileID, fileExt = data[1]:match("^(.+)/(.+)%.(.+)$");
    if (genresList[data[2]] == nil) then genresList[data[2]] = {fileID}; else genresList[data[2]][#genresList[data[2]]+1] = fileID; end
    genresHash[fileID] = data[2];
  end
  io.close(fIDGenre);
  fullDataset.genresList = genresList;
  fullDataset.genresHash = genresHash;
  #### ARTIST IMPORT ####-
  local curFile = 0;
  fIDArtist = io.open(artistFile, 'r');
  if (fIDArtist == nil) then print('  * Annotation file ' .. artistFile .. ' not found.'); return nil; end
  # Read the raw version
  for line in fIDArtist:lines('*l') do
    data = line:split("\t");
    curFile = curFile + 1;
    # These metadatas are full path indexed
    local path, fileID, fileExt = data[1]:match("^(.+)/(.+)%.(.+)$");
    if (artistsList[data[2]] == nil) then artistsList[data[2]] = {fileID}; else artistsList[data[2]][#artistsList[data[2]]+1] = fileID; end
    artistsHash[fileID] = data[2];
  end
  io.close(fIDArtist);
  fullDataset.artistsList = artistsList;
  fullDataset.artistsHash = artistsHash;
  #### SIMILARITY IMPORT ####-
  local curFile = 0;
  fIDSimilar = io.open(similarityFile, 'r');
  if (fIDSimilar == nil) then print('  * Annotation file ' .. similarityFile .. ' not found.'); return nil; end
  # Read the raw version
  for line in fIDSimilar:lines('*l') do
    data = line:split("\t");
    # Parse similar tracks
    if (#data > 2) then
      # Handle a more logical file hash
      if (filesHash[data[1]] ~= nil) then 
        curID = filesHash[data[1]];
      else
        curFile = curFile + 1;
        filesHash[data[1]] = curFile;
        filesList[curFile] = dataRoot .. '/data/mp3/' .. data[1] .. '.mp3';
        curID = curFile;
        hasSimilar[curFile] = 1;
        similarVector[#similarVector+1] = curFile;
      end
      curScores = torch.Tensor(#data - 1);
      curTracks = torch.Tensor(#data - 1);
      for t = 2,#data do
        curPair = data[t]:split(',');
        curScores[t-1] = tonumber(curPair[2]);
        if (filesHash[curPair[1]] ~= nil) then  
          tmpID = filesHash[curPair[1]];
        else
          curFile = curFile + 1;
          filesHash[curPair[1]] = curFile;
          filesList[curFile] = dataRoot .. '/data/mp3/' .. curPair[1] .. '.mp3';
          tmpID = curFile;
          hasSimilar[curFile] = 0;
        end
        curTracks[t-1] = tmpID;
      end
      filesSimilar[curID] = curTracks;
      filesScores[curID] = curScores;
    end
  end
  io.close(fIDSimilar);
  #### CREATE TRAIN / VALID ####-
  fullDataset.nbFiles = #filesList;
  fullDataset.filesList = filesList;
  fullDataset.filesHash = filesHash;
  fullDataset.filesSimilar = filesSimilar;
  fullDataset.filesScores = filesScores; 
  fullDataset.hasSimilar = torch.Tensor(hasSimilar);
  fullDataset.similarVector = torch.Tensor(similarVector);
  #### CREATE STRUCTURE ####-
  #randIDs = torch.randperm(fullDataset.similarVector:size(1));
  randIDs = fullDataset.similarVector;
  fullDataset.trainIDs = randIDs[{{1,math.floor(randIDs:size(1) * trainPercent)}}];
  fullDataset.validIDs = randIDs[{{math.floor(randIDs:size(1) * trainPercent)+1,randIDs:size(1)}}];
  fullDataset.isValid = torch.zeros(#filesList);
  for i = 1,fullDataset.validIDs:size(1) do
    fullDataset.isValid[fullDataset.validIDs[i]] = 1;
  end
  #### EXTRANEOUS DATASET INFO ####-
  fullDataset.loaded = torch.zeros(curFile);
  fullDataset.loadedTracks = 0;
  fullDataset.curEnd = 0;
  fullDataset.data = {};
  fullDataset.curBatch = {};
  fullDataset.nextBatch = {};
  fullDataset.toRemove = {};
  print('Found ' .. #fullDataset.filesList .. ' files.');
  return fullDataset;
end

"""

metadataCallbacks = {
  'artist':importRawLabel, 
  'beat':importTimedSeriesFile, 
  'downbeat':importTimedSeriesFile, 
  'emotion':importRawLabel, 
  'cover':importTrackList, 
  'genre':importRawLabel, 
  'instrument':importRawLabel,
  'key':importRawLabel, 
  'keys':importTimedSegmentFile, 
  'melody':importTimedNumbersFile, 
  'meter':importRawLabel, 
  'tempo':importRawNumber, 
  'tag':importRawMultiLabel,
  'multilabel':importRawMultiLabel, 
  'structure':importTimedSegmentFile,
  'harmony':importTimedSegmentFile,
#  'similarity':importSimilarity,
  'similarities':importTimedSegmentFile,
  'drum':importTimedLabelsFile,
  'chord':importTimedSegmentFile,
  'onset':importTimedSeriesFile,
  'year':importRawNumber,
  'gps':importNumberList,
  'location':importRawLabel,
  'tags':importLabelPairsList,
  'terms':importLabelPairsList,
  'review':importStringFile,
  'default':importRawLabel
};
