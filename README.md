## gp-net

[![DOI](https://zenodo.org/badge/276850997.svg)](https://zenodo.org/doi/10.5281/zenodo.13321075)

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
usage: gp-net.py -h

```

### Help
Please see the [wiki page](https://github.com/keeeto/gp-net/wiki) for description
of all the features of `gp-net`. If your questions are not answered in the wiki,
please contact us by email. If you have found a bug in any of the tools, please
[submit a ticket](https://github.com/keeeto/gp-net/issues) and we will attend to it. 


# 4. Citation

Cite this repo as follows:

```
@misc{gp-net,
title  = {Entropy-based active learning of graph neural network surrogate models for materials properties},
author = {Johannes Allotey, Keith T. Butler, Jeyan Thiyagalingam},
url    = {https://github.com/mdi-group/gp-net/},
doi    = {10.5281/zenodo.13321076}
year   = {2021}
 }
```

# 5. Acknowledgments

This work was partially supported by wave 1 of the UKRI Strategic Priorities Fund under the EPSRC (Grant No. EP/T001569/1), particularly the “AI for Science” theme within that grant and The Alan Turing Institute. The ML models were trained using computing resources provided by STFC Scientific Computing Department’s SCARF cluster and the PEARL cluster. We acknowledge support from STFC via the Data Intensive Centre for Doctoral Training (ST/P006779/1) and the University of Bristol.


