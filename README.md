## gp-net

`gp-net` is a regression tool for predicting the optical properties of materials,
and estimates the uncertainties on these predictions for the purpose
of active learning. 

### Features 
- Uncertainty Quantification 
  * Train-test split
  * k-fold cross-validation
- Pool-based sampling Active Learning
  * Entropy-based sampling 
  * Random-based sampling 

### Usage
```
usage: gp-net.py [-h] [-checkdata] [-ltype LTYPE] [-nomeg] [-noactive]
                 [-samp SAMP] [-cycle CYCLE CYCLE] [-repeat] [-q QUAN]
                 [-stop STOP] [-data DATA [DATA ...]] [-key KEY [KEY ...]]
                 [-frac FRAC [FRAC ...]] [-include] [-nsplit NSPLIT]
                 [-epochs EPOCHS] [-batch BATCH] [-bond BOND] [-nfeat NFEAT]
                 [-cutoff CUTOFF] [-width WIDTH] [-prev] [-layer LAYER]
                 [-ndims NDIMS] [-p PERP] [-niters NITERS] [-rate RATE]
                 [-amp AMP] [-length LENGTH]
                 [-maxiters MAXITERS [MAXITERS ...]]

Uncertainty quantification in neural networks.

optional arguments:
  -h, --help            show this help message and exit
  -checkdata            Check number of entries in the dataset. [default:
                        False]
  -ltype LTYPE          Display the layers in a fitted MEGNet model.
  -nomeg                Do not train with MEGNet. [default: False]
  -noactive             Don't do active learning [default: False]
  -samp SAMP            Type of sampling for active learning. Use random or
                        entropy [default: entropy]
  -cycle CYCLE CYCLE    Number of structures to sample and maximum number of
                        times to sample separated by spaces for active
                        learning. [default: 1 5]
  -repeat               MEGNet train and pre-process activations in each
                        active learning cycle [default: False]
  -q QUAN, --quan QUAN  Quantity of data for norepeat active learning
                        [default: 1000]
  -stop STOP            Maximum fraction of test set required for active
                        learning [default: 0.1]
  -data DATA [DATA ...]
                        Input dataset(s). Multiple datasets can be passed, one
                        per optical property of interest. [No default]
  -key KEY [KEY ...]    API key for data download and the optical properties
                        of interest, separated by spaces. For MEGNet users
                        only. [eg. Key band_gap formation_energy_per_atom
                        e_above_hull]
  -frac FRAC [FRAC ...]
                        Fraction of data for training and testing separated by
                        spaces for train-test split and k-fold cross-
                        validation. Fraction of data for training, and
                        fraction of training data for validation in repeat
                        active learning. For norepeat active learning, single
                        input as the fraction of the training data for
                        validation. [default: 0.3]
  -include              Include zero optical property values in the MEGNet
                        training and/or Gaussian process analysis. [default:
                        False]
  -nsplit NSPLIT        Number of training set splits for k-fold cross-
                        validation. [default: 1 i.e no cross-validation]
  -epochs EPOCHS        Epochs. [default: 0 ie. Perform no training with
                        MEGNet]
  -batch BATCH          Batch size for training with MEGNet or CNN. [default:
                        256]
  -bond BOND            MEGNet feature bond. [default: 10]
  -nfeat NFEAT          MEGNet feature global. [default: 2]
  -cutoff CUTOFF, --cutoff CUTOFF
                        MEGNet radial cutoff. [default: 5]
  -width WIDTH, --width WIDTH
                        MEGNet gaussian width. [default: 0.5]
  -prev                 Use a pre-trained MEGNet model during training with
                        MEGNet. [default: False]
  -layer LAYER          MEGNet fitted model layer to analyse. [default:
                        readout_0 i.e 32 dense layer]
  -ndims NDIMS          Dimensions of embedded space. 0 => Do not preprocess
                        activations, 1 => scale activations to 0, 1 range, 2
                        or 3 => Reduce dimensions of activations with tSNE.
                        [default: 0]
  -ndims NDIMS          Dimensions of embedded space. 0 => scale activations
                        in 0,1 range 2 or 3 => Reduce dimensions of
                        activations with tSNE. [default: 0]
  -p PERP, --perp PERP  Perplexity value to use in dimension reduction with
                        tSNE. [default: 150]
  -niters NITERS        Number of iterations for optimisation in tSNE.
                        [default: 1000]
  -rate RATE            Adam optimizer Learning rate. [default: 0.01]
  -amp AMP              Amplitude of the GP kernel. [default: 1.0]
  -length LENGTH        The length scale of the GP kernel. [default: 1.0]
  -maxiters MAXITERS [MAXITERS ...]
                        Maximum iterations for optimising GP hyperparameters.
                        For k-fold cross-validation, two inputs are required -
                        one for training per fold and the other for training
                        using train-test split. For active learning and train-
                        test split, a single input is required. [default: 0
                        i.e no GP training]

```

### Help
Please see the [wiki page](https://github.com/keeeto/gp-net/wiki) for description
of all the features of `gp-net`. If your questions are not answered in the wiki,
please contact us by email. If you have found a bug in any of the tools, please
[submit a ticket](https://github.com/keeeto/gp-net/issues) and we will attend to it. 


# 4. Citation

Cite this repo as follows:

```
@misc{injector:2021,
title  = {Injector-Surrogates: Machine Learning Surrogate Models for Particle Accelerator Injector Profile Simulation},
author = {Johannes Allotey, Keith T. Butler, Jeyan Thiyagalingam},
url    = {https://github.com/keeeto/gp-net/},
year   = {2021}
 }
```

# 5. Acknowledgments

This work was partially supported by wave 1 of the UKRI Strategic Priorities Fund under the EPSRC (Grant No. EP/T001569/1), particularly the “AI for Science” theme within that grant and The Alan Turing Institute. The ML models were trained using computing resources provided by STFC Scientific Computing Department’s SCARF cluster and the PEARL cluster. We acknowledge support from STFC via the Data Intensive Centre for Doctoral Training (ST/P006779/1) and the University of Bristol.


