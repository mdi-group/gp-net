"""
Passes the name of the optical property of interest
if the data is passed or data is downloaded from the
Materials Project if the API key is passed.

Other useful routines are accessible from here.
"""

import logging
import os

import pandas
import numpy

from megnet.data.graph import GaussianDistance
from megnet.data.graph import StructureGraph
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"), format="%(levelname)s:gp-net: %(message)s"
)


def load_data(data):
    """
    load_data(data)

    Load the passed datasets.

    Inputs:
        data-  Already downloaded dataset.

    Outputs:
         1-    List of requested material properties
               to be trained on.
    """
    props = []
    for dat in data:
        props.append(dat.split("_data.pkl")[0])
    return props


def read_data(datafile, keep_zeroes=False):
    """
    Checks the entries in the dataset so the user
    can decide on how to split data for processing.

    :param datafile: The data in .pkl format.
    :param keep_zeroes: Exclude/Include zero optical property values.

    :return: Number of entries in the dataset.
    """
    inputs = pandas.read_pickle(datafile)
    prop = datafile.split("_data")[0]
    print("\nNumber of input entries found for %s data = %s" % (prop, len(inputs)))
    if keep_zeroes:
        logging.info("Optical property values zero will not be excluded ...")
        structures = inputs["structure"].to_numpy()
        targets = inputs[prop].to_numpy()
        print("Remaining number of entries = %s" % len(targets))
    else:
        logging.info("Excluding zero optical property values from the dataset ...")
        mask = numpy.array([i for i, val in enumerate(inputs[prop]) if abs(val) == 0.0])
        structures = numpy.delete(inputs["structure"].to_numpy(), mask)
        targets = numpy.delete(inputs[prop].to_numpy(), mask)
        print("Remaining number of entries = %s" % len(targets))


def get_train_and_validation_sets(x_pool, y_pool, boundary):
    """
    Splits the pool set into training and validation sets.

    :param x_pool: Pool for the independent variable.
    :param y_pool: Pool for the dependent variable.
    :param boundary:

    :return:
    """
    x_train = x_pool[:-boundary]
    y_train = y_pool[:-boundary]

    x_val = x_pool[-boundary:]
    y_val = y_pool[-boundary:]

    return x_train, y_train, x_val, y_val


def _get_pool_boundary(valid_targets, pool_fraction):
    """
    Get the boundary for splitting the datasets for the pool

    :param valid_targets:
    :param pool_fraction:

    :return:
    """
    return int(len(valid_targets) * pool_fraction)


def megnet_input(prop, bond, nfeat_global, cutoff, width, keep_zeroes=False, *fraction):
    """
    Extracts valid structures and targets and splits them into user specified
    datasets.

    :param prop: Optical property of interest.
    :param bond: MEGNet feature bond.
    :param nfeat_global: MEGNet feature global.
    :param cutoff: MEGNet radial cutoff.
    :param width: MEGNet gaussian width.
    :param keep_zeroes: Include Exclude/Include zero optical property values.
    :param *fraction: Fraction of data to split into training and
                            validation sets. Passing an extra argument to
                            split data based on quantity is permissible.

    :return: Featurised structures for training, valid structures and targets,
        inputs for extraction of activations, pool, test, training, and validation
        sets.
    """
    logging.info("Get graph inputs to MEGNet ...")
    print("Bond features = ", bond)
    print("Global features = ", nfeat_global)
    print("Radial cutoff = ", cutoff)
    print("Gaussian width = ", width)
    gaussian_centers = numpy.linspace(0, cutoff, bond)
    distance_converter = GaussianDistance(gaussian_centers, width)
    graph_converter = CrystalGraph(bond_converter=distance_converter)
    model = MEGNetModel(bond, nfeat_global, graph_converter=graph_converter)

    datafile = "%s_data.pkl" % prop
    inputs = pandas.read_pickle(datafile)
    print("\nNumber of input entries found for %s data = %s" % (prop, len(inputs)))
    if keep_zeroes:
        logging.info("Zero optical property values will be included ...")
        structures = inputs["structure"].to_numpy()
        targets = inputs[prop].to_numpy()
    else:
        logging.info("Excluding zero optical property values from the dataset ...")
        mask = numpy.array([i for i, val in enumerate(inputs[prop]) if abs(val) == 0.0])
        structures = numpy.delete(inputs["structure"].to_numpy(), mask)
        targets = numpy.delete(inputs[prop].to_numpy(), mask)
        print("Remaining number of entries = %s" % len(targets))

    # Get the valid structures and targets i.e. exclude isolated atoms
    logging.info("Obtaining valid structures and targets ...")
    valid_structures = []
    valid_targets = []
    activations_input_full = []
    for s, t in zip(structures, targets):
        try:
            activations_input_full.append(StructureGraph.get_input(graph_converter, s))
        except:
            print("Skipping structure with isolated atom ...")
            continue
        valid_structures.append(s)
        valid_targets.append(t)
    print("Number of invalid structures = %s" % (len(targets) - len(valid_targets)))
    print("\nTotal number of entries available for analysis = %s" % len(valid_targets))

    pool_frac = fraction[0][0]
    pool_boundary = _get_pool_boundary(
        valid_targets, pool_frac
    )  # Data split is based on percentages

    if len(fraction) == 1:
        if (fraction[0][0] + fraction[0][1]) == 1.0:
            # For train-test split and k-fold cross-validation
            test_frac = fraction[0][1]

            logging.info("The pool is the same as the training set ...")
            print("Requested pool: %s%%" % (pool_frac * 100))
            print("Requested test set: %s%%" % (test_frac * 100))
            x_pool = numpy.array(valid_structures[0:pool_boundary])
            y_pool = numpy.array(valid_targets[0:pool_boundary])
            x_test = numpy.array(valid_structures[pool_boundary:])
            y_test = numpy.array(valid_targets[pool_boundary:])

            logging.info("The pool is the same as the training set ...")
            print("Pool:", y_pool.shape)
            print("Test set:", y_test.shape)

            return (
                model,
                activations_input_full,
                valid_structures,
                valid_targets,
                x_pool,
                y_pool,
                x_test,
                y_test,
            )

        elif (fraction[0][0] + fraction[0][1]) < 1.0:
            #  For repeat active learning
            val_frac = fraction[0][1]
            test_frac = numpy.round(1 - pool_frac, decimals=2)

            x_pool = numpy.array(valid_structures[0:pool_boundary])
            y_pool = numpy.array(valid_targets[0:pool_boundary])
            x_test = numpy.array(valid_structures[pool_boundary:])
            y_test = numpy.array(valid_targets[pool_boundary:])

            val_boundary = int(pool_boundary * val_frac)
            x_train, y_train, x_val, y_val = get_train_and_validation_sets(
                x_pool, y_pool, val_boundary
            )

            print("Requested validation set: %s%% of pool" % (val_frac * 100))
            print("Training set:", y_train.shape)
            print("Validation set:", y_val.shape)
            print("Test set:", y_test.shape)

            return (
                model,
                activations_input_full,
                valid_structures,
                valid_targets,
                x_pool,
                y_pool,
                x_test,
                y_test,
                x_train,
                y_train,
                x_val,
                y_val,
            )

    return (
        model,
        activations_input_full,
        numpy.array(valid_structures),
        numpy.array(valid_targets),
    )
