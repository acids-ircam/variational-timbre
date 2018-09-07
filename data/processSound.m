% Include the toolboxes
addpath(genpath('.'));
% First check the input arguments
if ~exist('resampleTo', 'var'), resampleTo = 22050; end
if ~exist('targetDuration', 'var'), targetDuration = 0; end
% Generic transforms arguments
if ~exist('winSize', 'var'), winSize = 2 ^ nextpow2(round(0.08 * resampleTo)); end
if ~exist('hopSize', 'var'), hopSize = round(0.2 * resampleTo); end
if ~exist('winSizeS', 'var'), winSizeS = (winSize / resampleTo); end
if ~exist('hopSizeS', 'var'), hopSizeS = (hopSize / resampleTo); end
if ~exist('winSizeMs', 'var'), winSizeMs = winSizeS * 1000; end
if ~exist('hopSizeMs', 'var'), hopSizeMs = hopSizeS * 1000; end
if ~exist('nFFT', 'var'), nFFT = 2 ^ nextpow2(winSize); end
% Mel-related parameters (cf. mel-gabor toolbox)
if ~exist('minFreq', 'var'), minFreq = 64; end
if ~exist('maxFreq', 'var'), maxFreq = 8000; end
if ~exist('nbBands', 'var'), nbBands = 128; end
% Mfcc-related (cf. mel-gabor toolbox)
if ~exist('nbCoeffs', 'var'), nbCoeffs = 13; end
if ~exist('delta', 'var'), delta = 0; end
if ~exist('dDelta', 'var'), dDelta = 0; end
% Gabor-related (cf. mel-gabor toolbox)
if ~exist('omegaMax', 'var'), omegaMax = [pi/2 pi/2]; end
if ~exist('sizeMax', 'var'), sizeMax = [3*nbBands 40]; end
if ~exist('nu', 'var'), nu = [3.5 3.5]; end
if ~exist('filterDistance', 'var'), filterDistance = [0.3 0.2]; end
if ~exist('filterPhases', 'var'), filterPhases = {[0 0], [0 pi/2], [pi/2 0], [pi/2 pi/2]}; end
% Chroma-related (cf. chroma toolbox)
if ~exist('chromaWinSize', 'var'), chromaWinSize = winSize; end
% CQT-related (cf. cqt toolbox)
if ~exist('cqtBins', 'var'), cqtBins = 24; end
if ~exist('cqtMin', 'var'), cqtFreqMin = minFreq; end
if ~exist('cqtMax', 'var'), cqtFreqMax = maxFreq; end
if ~exist('cqtGamma', 'var'), cqtGamma = 0.5; end
% Gammatone-related (cf. gammatone toolbox)
if ~exist('gammatoneBins', 'var'), gammatoneBins = 64; end
if ~exist('gammatoneMin', 'var'), gammatoneMin = minFreq; end
if ~exist('gammatoneMax', 'var'), gammatoneMax = maxFreq; end
% Wavelet-related (cf. scattering toolbox)
if ~exist('waveletType', 'var'), waveletType = 'gabor_1d'; end
if ~exist('waveletQ', 'var'), waveletQ = 8; end
% Scattering-related (cf. scattering toolbox)
if ~exist('scatteringDefault', 'var'), scatteringDefault = 1; end
if ~exist('scatteringTypes', 'var'), scatteringTypes = {'gabor_1d', 'morlet_1d', 'morlet_1d'}; end
if ~exist('scatteringQ', 'var'), scatteringQ = [8 2 1]; end
if ~exist('scatteringT', 'var'), scatteringT = 8192; end
% Cochleogram-related (cf. scattering toolbox)
if ~exist('cochleogramFrame', 'var'), cochleogramFrame = 64; end        % frame length, typically, 8, 16 or 2^[natural #] ms.
if ~exist('cochleogramTC', 'var'), cochleogramTC = 16; end              % time const. (4, 16, or 64 ms), if tc == 0, the leaky integration turns to short-term avg.
if ~exist('cochleogramFac', 'var'), cochleogramFac = -1; end            % nonlinear factor (typically, .1 with [0 full compression] and [-1 half-wave rectifier]
if ~exist('cochleogramShift', 'var'), cochleogramShift = 0; end         % shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
if ~exist('cochleogramFilter', 'var'), cochleogramFilter = 'p'; end     % filter type ('p' = Powen's IIR, 'p_o' = steeper group delay)
% STRF-related (cf. auditory toolbox)
if ~exist('strfFullT', 'var'), strfFullT = 0; end                       % fullT (fullX): fullness of temporal (spectral) margin in [0, 1].
if ~exist('strfFullX', 'var'), strfFullX = 0; end 
if ~exist('strfBP', 'var'), strfBP = 0; end                             % Pure Band-Pass indicator
if ~exist('strfRv', 'var'), strfRv = 2 .^ (1:.5:5); end                 % rv: rate vector in Hz, e.g., 2.^(1:.5:5).
if ~exist('strfSv', 'var'), strfSv = 2 .^ (-2:.5:3); end                % scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
if ~exist('strfMean', 'var'), strfMean = 0; end                         % Only produce the mean activations
% Normalization parameters
if ~exist('normalizeInput', 'var'), normalizeInput = 0; end
if ~exist('normalizeOutput', 'var'), normalizeOutput = 0; end
if ~exist('equalizeHistogram', 'var'), equalizeHistogram = 0; end
if ~exist('logAmplitude', 'var'), logAmplitude = 1; end
% Phase-related parameters
if ~exist('removePhase', 'var'), removePhase = 1; end
if ~exist('concatenatePhase', 'var'), concatenatePhase = 0; end
% Debug mode
if ~exist('debugMode', 'var'), debugMode = 0; end
% Create the analysis folder
if ~exist(newRoot, 'dir'), mkdir(newRoot); end
% Check for the type of analysis or do all
%if ~(exist('transformType', 'var')), transformType = {'stft'}; end
% If the transform type "all" is asked, fill transform with all known ones
if (ismember('all', transformType))
    transformType = {'stft', 'mel', 'mfcc', 'gabor', 'chroma', 'cqt', 'gammatone', 'dct', 'hartley', 'rasta', 'plp', 'wavelet', 'scattering', 'cochleogram', 'strf', 'modulation'};
end
                                                                                       disp(transformType);
% Load the cortical toolbox (only if required)
if (ismember('cochleogram', transformType) || ismember('strf', transformType)), loadload; end
% Perform pre-processing of the scattering 
if (ismember('wavelet', transformType))
    waveletopt = default_filter_options('audio', winSize);
    waveletFilt = wavelet_factory_1d(targetDuration * resampleTo, waveletopt);
end
% Perform pre-processing of the scattering 
if (ismember('scattering', transformType))
    if (scatteringDefault)
        filt_opt = default_filter_options('audio', winSize);
    else
        filt_opt = struct;
        filt_opt.filter_type = scatteringTypes;
        filt_opt.Q = scatteringQ;
        filt_opt.J = T_to_J(scatteringT * 2, filt_opt);
    end
    Wop = wavelet_factory_1d(targetDuration * resampleTo, filt_opt);
end
% Parse through the mini-batch
for i = 1:length(audioList)
    % Current audio file
    audioFile = audioList{i};
    % Obtain full path informations
    [path, file, ext] = fileparts(audioFile);
    % Path to the analysis file
    if (~exist('oldRoot', 'var') || isempty(oldRoot)), newPath = newRoot; else newPath = regexprep(path, oldRoot, newRoot); end
    % Check if transforms already computed
    if (exist([newPath '/' file '.mat'], 'file'))
        continue;
    end
    % Special case for MP3 files
    if strcmpi(ext, '.mp3')
        [signalI, sRate] = mp3read(audioFile);
        audiowrite('/tmp/tempCorticalFile.wav', signalI, sRate);
        audioFile = '/tmp/tempCorticalFile.wav';
    end
    % Read the corresponding signal
    disp(['  * Processing ' audioFile])
    [sig, fs] = audioread(audioFile);
    % Turn to mono if multi-channel file
    if size(sig, 2) > 1, sig = mean(sig, 2); end
    % First resample the signal (similar to ircamDescriptor)
    if (fs ~= resampleTo)
        % Be a bit clever and limit the amount of ops
        gcdVal = gcd(fs, resampleTo);
        % Resample the signal
        sig = resample(sig, resampleTo / gcdVal, fs / gcdVal);
        fs = resampleTo;
    end
    % Now ensure that we have the target duration
    if (targetDuration)
        % Input is longer than required duration
        if ((length(sig) / fs) > targetDuration)
            % Select the center of the sound and trim around
            midPoint = floor(length(sig) / 2); midLen = floor((targetDuration * fs) / 2) + 1;
            sPoint = midPoint - midLen; ePoint = midPoint + midLen;
            sig = sig(sPoint:ePoint);
        end
        % Otherwise pad with zeros
        if (length(sig) / fs) < targetDuration
            sig = padarray(sig, floor(((targetDuration * fs) - length(sig)) / 2) + 1);
        end
    end
    % Check if we need to normalize the input
    if (normalizeInput)
        sig = sig ./ max(sig);
    end
    % Final transform structure
    transforms = {};
    for t = 1:length(transformType) 
        % Type of transform
        currentType = transformType{t};
        % Compute transform
        switch (currentType)
            % Compute the FFT
            case 'stft'
                currentTransform = spectrogram(sig, winSize, (winSize - hopSize), nFFT, resampleTo);
            case 'mel'
                currentTransform = log_mel_spectrogram(sig, resampleTo, hopSizeMs, winSizeMs, [minFreq maxFreq], nbBands);
            case 'mfcc'
                melSpec = log_mel_spectrogram(sig, resampleTo, hopSizeMs, winSizeMs, [minFreq maxFreq], nbBands);
                currentTransform = mfcc(melSpec, nbCoeffs, delta, dDelta);
            case 'gabor'
                melSpec = log_mel_spectrogram(sig, resampleTo, hopSizeMs, winSizeMs, [minFreq maxFreq], nbBands);
                currentTransform = sgbfb(melSpec, omegaMax, sizeMax, nu, filterDistance, filterPhases);
            case 'chroma'
                currentTransform = chromagram_IF(sig, resampleTo, chromaWinSize);
            case 'cqt'
                currentTransform = cqtTrans(sig, cqtBins, resampleTo, cqtFreqMin, cqtFreqMax, hopSize, 'gamma', cqtGamma);
                currentTransform = currentTransform.d;
	    case 'cqt2'
		[currentTransform,~,~] = cqtTransform(sig, nFFT, hopSize, resampleTo, Q);
            case 'gammatone'
                currentTransform = gammatonegram(sig, resampleTo, winSizeS, hopSizeS, gammatoneBins, gammatoneMin, gammatoneMax);
            case 'dct'
                currentTransform = myDCT(sig, resampleTo, winSize, hopSize, nFFT);
            case 'hartley'
                currentTransform = myHartley(sig, resampleTo, winSize, hopSize, nFFT);
            case 'rasta'
                [~, currentTransform] = rastaplp(sig, resampleTo);
            case 'plp'
                [~, currentTransform] = rastaplp(sig, resampleTo, 0, 12);
            case 'wavelet'
                if (targetDuration == 0), waveletFilt = wavelet_factory_1d(length(sig), waveletopt); end
                S = scat(sig, waveletFilt); S = renorm_scat(S); S = log_scat(S);
                currentTransform = cell2mat(S{2}.signal)';
            case 'scattering'
                if (targetDuration == 0), Wop = wavelet_factory_1d(length(sig), filt_opt); end
                S = scat(sig, Wop);
                S = renorm_scat(S); S = log_scat(S);
                currentTransform = [cell2mat(S{1}.signal)' ; cell2mat(S{2}.signal)' ; cell2mat(S{3}.signal)'];
            case 'cochleogram'
                currentTransform = wav2aud(sig, [cochleogramFrame, cochleogramTC, cochleogramFac, cochleogramShift], cochleogramFilter, 0)';
            case 'strf'
                y = wav2aud(sig, [cochleogramFrame, cochleogramTC, cochleogramFac, cochleogramShift], cochleogramFilter, 0);
                currentTransform = aud2cor(y, [[cochleogramFrame, cochleogramTC, cochleogramFac, cochleogramShift] strfFullT strfFullX strfBP], strfRv, strfSv, 'tmpxxx', 0);
                if (strfMean), currentTransform = reshape(abs(mean(mean(currentTransform(:, :, :, :), 1), 2)), size(currentTransform, 3), size(currentTransform, 4))'; end
            case 'modulation'
                currentTransform = modspect(sig, fs, 'mTap');
                disp(size(currentTransform));
            case 'descriptors'
                currentTransform = processDescriptor(audioFile);
            otherwise
                error(['Unknown transform ' currentType]);
        end
        % Remove phase
        if (removePhase), currentTransform = abs(currentTransform); end
        % Concatenate the phase with the real part
        if (concatenatePhase), currentTransform = [abs(currentTransform); angle(currentTransform)]; end
        % Normalize output
        if (normalizeOutput), currentTransform = currentTransform ./ max(max(currentTransform)); end
        % Put in log-amplitude scale
        if (logAmplitude), currentTransform = log1p(currentTransform); end
        % Equalize histogram
        if (equalizeHistogram)
            if (isreal(currentTransform)) 
                currentTransform = heq(currentTransform);
            else
                currentTransform = heq(abs(currentTransform));
            end
        end
        % Plot the current transform if debugging
        if (debugMode), 
            if (length(size(currentTransform)) == 2)
                figure; imagesc(flipud(abs(currentTransform))); 
            else
                figure; imagesc(reshape(abs(sum(sum(currentTransform(:, :, :, :), 1), 2)), size(currentTransform, 3), size(currentTransform, 4)));
            end
        end
        % Finally store the transform
        transforms.(currentType) = single(currentTransform);
    end
    % Create path if inexistent
    if ~exist(newPath, 'dir')
        mkdir(newPath);
    end
    % Finally save all the transforms
    save([newPath '/' file '.mat'], 'transforms');
end
