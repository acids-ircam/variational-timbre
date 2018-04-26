# Generative variational timbre spaces

This repository describes the additional material and experiments around "Generative timbre spaces with variational audio synthesis" submitted at DaFX 2018.

For a better viewing experience of these additional results, please **visit the [corresponding Github pages website](https://acids-ircam.github.io/variational-timbre/ "DaFX 2018 - Latent spaces")**

This [supporting page](https://acids-ircam.github.io/variational-timbre/ "DaFX 2018 - Latent spaces") directly embeds the exposed elements
  * Supplementary figures
  * Animations of descriptor space traversal (topology)
  * Audio examples of synthesized paths in the space
  * Further detailed space for perceptual inference
  * Additional data and experiments
  
Otherwise, you can directly parse through the different sub-folders of the main `docs/` folder.

## Code

**The full code will only be released upon acceptance of the paper at the DaFX 2018 conference.**

### Dependencies

The code has been developed on Python3.5, it should work with other versions of Python, but this have not been tested. We rely on several libraries for different aspects of the code. The complete list is `numpy`, `scipy`, `pytorch`, `matplotlib`, `scikit-learn`, `nsgt`, `scikit-image` and `pyro`

Here is an helper list of pip3 install commands to facilitate your install

```
pip3 install numpy
pip3 install scipy
pip3 install pytorch
pip3 install matplotlib
pip3 install scikit-learn
pip3 install scikit-image
pip3 install pyro
pip3 install nsgt
```

The code can also work on GPUs, in which case you need to add the following dependencies (based on the premises that you have a working GPU with CUDA installed)

### Usage

The code is mostly divided into two scripts `dafx2018learn.py` and `dafx2018figures.py`. The first script `dafx2018learn.py` allows to train a model from scratch as described in the paper. This model can be regularized or not on the timbre space. The second script `dafx2018figures.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on the [supporting page](https://acids-ircam.github.io/variational-timbre/ "DaFX 2018 - Latent spaces") of this repository.

In both cases, the scripts can be used with the following options

```
Analysis arguments
  --dbroot           Root folder towards database
  --analysisroot     Target folder to find the analyzed transforms
  --dbname           Name of the database
  --sr               Sample rate
Representation-related arguments
  --transform        Representation to use (default=nsgt-cqt)
  --log              Use log amplitude [0 = No, 1 = Log, 2 = Tanh]
  --downsample       Downsample representation
  --preprocess       Pre-process representation [0 = No, 1 = PCA, 2 = MPCA]
  --nframes          Number of temporal frames given to the VAE
Model-related arguments
  --model            Type of model to use [vanilla, dlgm]
  --regularization   Type of regularization [none, prior, l2, gaussian, student]
  --mds_dims         Number of dimensions for perceptual multi-dimensional scaling
  --latent_dims      Number of latent dimensions in vae
  --beta             Beta weight in loss
  --alpha            Alpha weight of regularization
  --warmup           Number of epochs to use as warmup of beta value
  --normalize        Normalize the regularization distance
Properties of the learning
  --epochs           Maximum number of epochs
  --batchsize        Size of the batch
  --units            Number of hidden units
  --layers           Number of hidden layers
# Twofold learning
  --twofold          Restart learning from a previous model
  --targetdims       Which dimensions to regularize ['all', 'first', 'maxvar', 'minvar']
# GPU option
  --cuda             Index of GPU to use (-1 = CPU learning)

```

#### Example runs

1. Running the unregularized model to learn a VAE space of musical instruments

```
python3 dafx2018learn.py --regularization none --beta 2 --warmup 100 --units 2000
```

2. Learning a L2-regularized VAE space of musical instruments based on timbre distances

```
python3 dafx2018learn.py --regularization l2 --beta 2 --warmup 100 --units 2000 --alpha 1 --normalize 1
```
