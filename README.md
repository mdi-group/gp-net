## gp-net

`gp-net` is a tool for estimating the uncertainties on the
predicted properties of materials for the purpose of
active learning. 

### Features 
- Uncertainty Quantification 
  * [Train-test split](https://github.com/keeeto/gp-net/wiki/Train-test-split)
  * [k-fold cross-validation](https://github.com/keeeto/gp-net/wiki/k-fold-cross-validation)
- [Pool-based sampling Active Learning](https://github.com/keeeto/gp-net/wiki/Active-learning)
  * Entropy-based sampling 
  * Random-based sampling 

### Usage
```
usage: gp-net.py [-h] [-checkdata] [-ltype LTYPE] [-nomeg] [-noactive]
                 [-samp SAMP] [-cycle CYCLE CYCLE] [-norepeat] [-q QUAN]
                 [-stop STOP] [-data DATA [DATA ...]] [-key KEY [KEY ...]]
                 [-frac FRAC FRAC] [-include] [-nsplit NSPLIT]
                 [-epochs EPOCHS] [-batch BATCH] [-bond BOND] [-nfeat NFEAT]
                 [-cutoff CUTOFF] [-width WIDTH] [-prev] [-mlayer MLAYER]
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
                        learning. [default: 20 5]
  -norepeat             Do not train with MEGNet in each active learning cycle
                        [default: False]
  -q QUAN, --quan QUAN  Quantity of data for norepeat active learning [No
                        default]
  -stop STOP            Maximum fraction of test set required for active
                        learning [default: 0.1]
  -data DATA [DATA ...]
                        Input dataset(s). Multiple datasets can be passed, one
                        per optical property of interest. [No default]
  -key KEY [KEY ...]    API key for data download and the optical properties
                        of interest, separated by spaces. For MEGNet users
                        only. [eg. Key band_gap formation_energy_per_atom
                        e_above_hull]
  -frac FRAC FRAC       Fraction of data for training and fraction of the
                        training set for validation separated by spaces.
                        [default: 0.3 0.7]			
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
  -mlayer MLAYER        MEGNet fitted model layer to analyse. [default:
                        readout_0 i.e 32 dense layer]
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
                        using train-test split. For active learning and no
                        k-fold cross-validation, a single input is required.
                        [default: 0 0 i.e no MEGNet and GP training]			

```

### Help
Please see the [wiki page](https://github.com/keeeto/gp-net/wiki) for description
of all the features of `gp-net`. If your questions are not answered in the wiki,
please contact us by email. If you have found a bug in any of the tools, please
[submit a ticket](https://github.com/keeeto/gp-net/issues) and we will attend to it. 
