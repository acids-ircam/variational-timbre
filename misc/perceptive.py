import numpy as np, os, pdb
from scipy.stats import norm
from visualize.visualize_dimred import MDS


equivalenceInstruments = ['Clarinet-Bb', 'Alto-Sax', 'Trumpet-C', 'Violoncello', 
                          'French-Horn', 'Oboe', 'Flute', 'English-Horn', 
                          'Bassoon', 'Tenor-Trombone', 'Piano', 'Violin']


def get_perceptual_centroids(dataset, mds_dims, timbre_path='timnre.npy', covariance=True, timbreNormalize=True, timbreProcessing=True):
    if (timbreProcessing == True or (not os.path.isfile('timbre_' + str(mds_dims) + '.npy'))):
        fullTimbreData = np.load(timbre_path).item()
        # Names of the pre-extracted set of instruments (all with pairwise rates)
        selectedInstruments = fullTimbreData['instruments']
        # Full sets of ratings (i, j) = all ratings for instru. i vs. instru. j
        detailedMatrix = fullTimbreData['ratings']
        # Final matrices
        nbIns = len(selectedInstruments)
        meanRatings = np.zeros((nbIns, nbIns))
        gaussMuRatings = np.zeros((nbIns, nbIns))
        gaussStdRatings = np.zeros((nbIns, nbIns))
        nbRatings = np.zeros((nbIns, nbIns))
        # Fit Gaussians for each of the sets of pairwise instruments ratings
        for i in range(nbIns):
            for j in range(i+1, nbIns):
                nbRatings[i, j] = detailedMatrix[i, j].size
                meanRatings[i, j] = np.mean(detailedMatrix[i, j])
                # Model the gaussian distribution of ratings
                mu, std = norm.fit(detailedMatrix[i, j])
                # Fill parameters of the Gaussian        
                gaussMuRatings[i, j] = mu
                gaussStdRatings[i, j] = std
                print("%s vs. %s : mu = %.2f,  std = %.2f" % (selectedInstruments[i], selectedInstruments[j], mu, std))
        # Create square matrices
        meanRatings += meanRatings.T   
        gaussMuRatings += gaussMuRatings.T
        gaussStdRatings += gaussStdRatings.T
        meanRatings = (meanRatings - np.min(meanRatings)) / np.max(meanRatings)
        # Rescale means 
        gaussMuRatings = (gaussMuRatings - np.min(gaussMuRatings)) / np.max(gaussMuRatings)
        # Rescale variances
        gaussStdRatings = (gaussStdRatings - np.min(gaussStdRatings)) / np.max(gaussStdRatings)
        variance = np.mean(gaussStdRatings, axis=1)
        if (timbreNormalize):
            variance = ((variance - (np.min(variance)) + 0.01) / np.max(variance)) * 2
        # Compute MDS on Gaussian mean
        seed = np.random.RandomState(seed=3)        
        mds = MDS(n_components=mds_dims, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
        position = mds.fit(gaussMuRatings).embedding_
        # Store all computations here
        fullTimbreData = {'instruments':selectedInstruments, 
                          'ratings':detailedMatrix,
                          'gmean':gaussMuRatings,
                          'gstd':gaussStdRatings,
                          'pos':position,
                          'var':variance}
        np.save('timbre_' + str(mds_dims) + '.npy', fullTimbreData)
    else:
        # Retrieve final data structure
        fullTimbreData = np.load('timbre.npy').item()
        # Names of the pre-extracted set of instruments (all with pairwise rates)
        selectedInstruments = fullTimbreData['instruments']
        # Gaussian modelization of the ratings
        gaussMuRatings = fullTimbreData['gmean']
        gaussStdRatings = fullTimbreData['gstd']
        # MDS modelization of the ratings 
        position = fullTimbreData['pos']
        variance = fullTimbreData['var']
        
    audioTimbreIDs = np.zeros(len(equivalenceInstruments)).astype('int')
    
    # Parse through the list of instruments
    for k, v in dataset.classes['instrument'].items():
        if (k != '_length'):
            audioTimbreIDs[v] = equivalenceInstruments.index(k)
    # Class-dependent means and covariances
    prior_mean = position[audioTimbreIDs]
    prior_std = np.ones((len(equivalenceInstruments), mds_dims))
    if (covariance == 1):
        prior_std = prior_std * variance[audioTimbreIDs, np.newaxis]
    prior_params = (prior_mean, prior_std)
    # Same for full Gaussian
    prior_gauss_params = (gaussMuRatings, gaussStdRatings)  

    return prior_params, prior_gauss_params
        

    