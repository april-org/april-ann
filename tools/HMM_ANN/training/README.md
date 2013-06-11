HMM_ANN/train directory: training of HMM/ANNs
=============================================

The most important script here is `train.lua` script, which is prepared to train
HMM/ANNs following previous described algorithm. This wiki is a mere description
of the scripts. Currently non tutorial is available, but you can follow HTK
tutorial in order to understand the meaning of all options presented
here. Besides, ANN basic knowledge will be helpful.

The script receives a lot of parameters which are shown executing `april-ann
train.lua --help`:

```
USAGE:
	train.lua [-fVALUE] --train-m=VALUE --train-f=VALUE [--train-s=VALUE]
                  [--train-phdict=VALUE] --val-m=VALUE --val-f=VALUE
                  [--val-s=VALUE] [--val-phdict=VALUE] [--begin-sil=VALUE]
                  [--end-sil=VALUE] [--count-values=VALUE] --num-states=VALUE
                  --h1=VALUE --h2=VALUE -nVALUE [--train-r=VALUE]
                  [--val-r=VALUE] --tiedfile=VALUE [--context=VALUE]
                  [--feats-format=VALUE] [--feats-norm=VALUE] [--step=VALUE]
                  [--mean=VALUE] [--var=VALUE] [--seedp=VALUE] [--firstlr=VALUE]
                  [--epochs-firstlr=VALUE] [--lr=VALUE] [--mt=VALUE] [--wd=VALUE]
                  [--rndw=VALUE] [--seed1=VALUE] [--seed2=VALUE] [--seed3=VALUE]
                  [--epochs-wo-val=VALUE] [--epochs-wo-imp=VALUE]
                  [--epochs-wo-exp=VALUE] [--epochs-max=VALUE]
                  [--epochs-first-max=VALUE] [--em-it=VALUE]
                  [--initial-mlp=VALUE] [--initial-hmm=VALUE]
                  [--initial-em-epoch=VALUE] [--transcription-filter=VALUE]
                  [--silences=VALUE] [(-h|--help)] 

DESCRIPTION:
	HMM/ANN training with April-ANN toolkit

	-fVALUE	      Load configuration file (a lua tabla) (optional)
	--train-m=VALUE	      MFCC file for training set
	--train-f=VALUE	      FON file for training set
	--train-s=VALUE	      Initial segmentation file for training set (optional)
	--train-phdict=VALUE	      Phonetical dictionary for training (optional)
	--val-m=VALUE	      MFCC file for validation set
	--val-f=VALUE	      FON file for validation set
	--val-s=VALUE	      Initial segmentation file for validation set (optional)
	--val-phdict=VALUE	      Phonetical dictionary for validation (optional)
	--begin-sil=VALUE	      Initial silence (optional)
	--end-sil=VALUE	      Final silence (optional)
	--count-values=VALUE	       (optional) [DEFAULT: ]
	--num-states=VALUE	      Number of states per each HMM [DEFAULT: false]
	--h1=VALUE	      First hidden layer size
	--h2=VALUE	      Second hidden layer size
	-nVALUE	      Number of parameters per each frame
	--train-r=VALUE	      Replacement for training (0 for disable) (optional) [DEFAULT: 300000]
	--val-r=VALUE	      Replacement for validation (0 for disable) (optional) [DEFAULT: 0]
	--tiedfile=VALUE	      HTK unit's tied list
	--context=VALUE	      Size of ann context (optional) [DEFAULT: 4]
	--feats-format=VALUE	      Format of features mat or mfc or png (mat, png or mfc) (optional) [DEFAULT: mat]
	--feats-norm=VALUE	      Table with means and devs for features (optional)
	--step=VALUE	      Dataset step (optional) [DEFAULT: 1]
	--mean=VALUE	      Mean of gaussian perturbation (default 0) (optional) [DEFAULT: 0]
	--var=VALUE	      Variance of gaussian perturbation (0 for disable) (optional) [DEFAULT: 0.015]
	--seedp=VALUE	      Perturbation seed (optional) [DEFAULT: 86544]
	--firstlr=VALUE	      First learning rate (optional) [DEFAULT: 0.005]
	--epochs-firstlr=VALUE	      Num epochs for first learning rate (optional) [DEFAULT: 100]
	--lr=VALUE	      Learning rate (optional) [DEFAULT: 0.001]
	--mt=VALUE	      Momentum (optional) [DEFAULT: 0.005]
	--wd=VALUE	      Weight decay (optional) [DEFAULT: 1e-06]
	--rndw=VALUE	      Size of the random inf/sup for MLP (optional) [DEFAULT: 0.1]
	--seed1=VALUE	      Seed 1 (optional) [DEFAULT: 1234]
	--seed2=VALUE	      Seed 2 (optional) [DEFAULT: 4567]
	--seed3=VALUE	      Seed 3 (optional) [DEFAULT: 9876]
	--epochs-wo-val=VALUE	      Epochs without validation (optional) [DEFAULT: 4]
	--epochs-wo-imp=VALUE	      Epochs without improvement (optional) [DEFAULT: 20]
	--epochs-wo-exp=VALUE	      Epochs without expectation step (optional) [DEFAULT: 5]
	--epochs-max=VALUE	      Number of epochs for maximization step (optional) [DEFAULT: 100]
	--epochs-first-max=VALUE	      Number of epochs for first maximization step (optional) [DEFAULT: 500]
	--em-it=VALUE	      Number of EM iterations (default 100) (optional) [DEFAULT: 100]
	--initial-mlp=VALUE	      Initial MLP for continue an stopped training (optional)
	--initial-hmm=VALUE	      Initial HMM for continue an stopped training, or to use as a first HMM description (optional)
	--initial-em-epoch=VALUE	      Initial EM epoch for continue an stopped training (optional)
	--transcription-filter=VALUE	      Filter the transcriptions to generate phonetic sequences (optional)
	--silences=VALUE	      HMM models for silences (a blank separated list) (optional) [DEFAULT: ]
	-h, --help	      shows this help message (optional)
```

There are a lot of options to configure the training.

In the following, when we talk about **files list**, they are files which
contains a list of filenames, one per line (the files could be gzipped):

```
corpus/n02-000-00.mat.gz
corpus/n02-000-01.mat.gz
corpus/n02-000-02.mat.gz
```

The most important options are those which are not marked as `(optional)`. 

- `train-m`: A files list which contains the parametrization used as input. 
This list is used for model parameter estimation (training).

- `train-f`: A files list which contains textual transcriptions of parametrized
files contained in previous option. This list is used for model parameter
estimation (training). If this file is not a phonetic sequence, the parameter
`transcription-filter` is needed to convert the textual sequence in phonemes.
Otherwise, the option `train-phdict` could replace the transcription filter.

- `val-m`: Idem as `train-m` but for for early stopping (validation).

- `val-f`: Idem as `train-f` but for for early stopping (validation).

_ `h1`: Number of neurons at the first layer.

- `h2`: Number of neurons at the second layer (zero is valid, only one layer
  will be used).

- `n`: Number of parameters on each frame.

- `num-states`: Number of states per each phoneme.

- `tiedfile`: A list of valid phonemes, following HTK tied-list.
