"""A tool for quantifying uncertainties in a Graph Neural Network"""

import argparse
import sys
import logging
import os
import subprocess

import numpy as np
from sklearn.model_selection import KFold

from aux.pool_sampling import SelectionFunction
from aux.get_info import load_data, megnet_input
from aux.activations import latent
from aux.plotting import plot

from train.MEGNetTrain import training
from optimisers.adam import adam

log_level = os.getenv("LOG_LEVEL", "INFO")
log = logging.getLogger("gp-net")
log.setLevel(getattr(logging, log_level.upper()))

EntropySelection = SelectionFunction.entropy_based
RandomSelection = SelectionFunction.random_based


def _check_args(args):
    """
    Checks the arguments passed.

    :param args: The arguments passed from the command line.

    :return: None
    """
    if args.noactive:
        log.info("No active learning requested...")
        if len(args.frac) != 2:
            raise ValueError("--frac requires two inputs!")
        if not np.isclose(args.frac[0] + args.frac[1], 1.0):
            raise ValueError("The sum of --frac must be 1!")
        if not (0 < args.frac[0] and args.frac[1] < 1):
            raise ValueError("--frac values must be between 0 and 1!")
        if args.nsplit == 1:
            log.info("Train-test split approach requested...")
            if len(args.maxiters) > 1:
                raise ValueError("--maxiters must have length 1!")
            # maxiters = args.maxiters[0]
        else:
            print("%s-fold cross-validation requested..." % args.nsplit)
            if len(args.maxiters) != 2:
                raise ValueError("--maxiters must have length 2!")

    else:
        logging.info("Requested to perform active learning...")
        if args.stop >= 1.0:
            raise ValueError("--stop argument should be less than 1 but not zero!")
        if args.nsplit != 1:
            raise ValueError(
                "Active learning with k-fold cross validation not supported!"
            )
        if len(args.maxiters) != 1:
            raise ValueError("--maxiters must have length 1!")
        # maxiters = args.maxiters[0]
        if args.repeat:
            log.info(
                "MEGNet train and perform activation analysis per cycle of active learning..."
            )
            if len(args.frac) != 2:
                raise ValueError("--frac requires two inputs!")
            if not (args.frac[0] + args.frac[1]) < 1.0:
                raise ValueError("The sum of --frac must be less than 1!")
            if not (0 < args.frac[0] and args.frac[1] < 1):
                raise ValueError("--frac values must be between 0 and 1!")
        else:
            log.info(
                "MEGNet train and perform activation analysis ONCE during the active learning ..."
            )
            if len(args.frac) != 1:
                raise ValueError(
                    "--frac requires a single input as the validation fraction!"
                )
            if 1.0 > args.frac[0] > 0.0:
                raise ValueError(
                    "--frac must be less than 1!"
                )  # fraction is a list, and we need to pass a single input
            if not args.quan:
                raise ValueError("Provide quantity of data to use with --quan!")


def main():
    """From command line, all parsing are handled here"""
    parser = argparse.ArgumentParser(
        prog="gp-net", description="Uncertainty quantification in Neural networks."
    )
    group1 = parser.add_argument_group("Data and Data Split Options")
    group1.add_argument(
        "--data",
        help="Input dataset(s). Multiple datasets can be passed, one per optical property "
        "of interest. [No default]",
        type=str,
        nargs="+",
        required=True,
    )
    group1.add_argument(
        "--frac",
        help="Fraction of data for training and testing separated by spaces for train-test "
        "split and k-fold cross-validation. Fraction of data for training, and fraction "
        "of training data for validation in repeat active learning. For norepeat active "
        "learning, single input as the fraction of the training data for validation. "
        "[default: 0.3]",
        nargs="+",
        type=float,
        default=[0.3],
    )
    group1.add_argument(
        "--include",
        action="store_true",
        help="Include zero optical property values in the MEGNet training and/or Gaussian Process "
        "analysis. [default: False]",
        default=False,
    )
    group1.add_argument(
        "--nsplit",
        help="Number of training set splits for k-fold cross-validation. [default: 1 i.e no cross-validation]",
        type=int,
        default=1,
    )

    group2 = parser.add_argument_group("Training with MEGNet Options")
    group2.add_argument(
        "--nomeg",
        action="store_true",
        help="Do not train with MEGNet. [default: False]",
        default=False,
    )
    group2.add_argument(
        "--epoch",
        help="Epoch. [default: 0 ie. Perform no training with MEGNet]",
        type=int,
        default=0,
    )
    group2.add_argument(
        "--batch",
        help="Batch size for training with MEGNet or CNN. [default: 256]",
        type=int,
        default=256,
    )
    group2.add_argument(
        "--bond", help="MEGNet feature bond. [default: 10]", type=int, default=10
    )
    group2.add_argument(
        "--nfeat", help="MEGNet feature global. [default: 2]", type=int, default=2
    )
    group2.add_argument(
        "--rcut",
        help="MEGNet radial cutoff. [default: 5]",
        type=int,
        default=5,
    )
    group2.add_argument(
        "--width",
        help="MEGNet gaussian width. [default: 0.5]",
        type=float,
        default=0.5,
    )
    group2.add_argument(
        "--prev",
        action="store_true",
        help="Use a pre-trained MEGNet model during training with MEGNet. [default: False]",
        default=False,
    )
    group2.add_argument(
        "--layer",
        help="MEGNet fitted model layer to analyse. [default: readout_0 i.e 32 dense layer]",
        type=str,
        default="readout_0",
    )

    group3 = parser.add_argument_group("Dimensionality Reduction Options")
    group3.add_argument(
        "--ndims",
        help="Dimensions of embedded space. 0 => Do not preprocess activations, 1 => scale "
        "activations to 0, 1 range, 2 or 3 => Reduce dimensions of activations with tSNE. "
        "[default: 0]",
        type=int,
        default=0,
    )
    group3.add_argument(
        "--perp",
        help="Perplexity value to use in dimension reduction with tSNE. [default: 150]",
        type=float,
        default=150,
    )
    group3.add_argument(
        "--niters",
        help="Number of iterations for optimisation in tSNE. [default: 1000]",
        type=int,
        default=1000,
    )

    group4 = parser.add_argument_group("Gaussian Process Options")
    group4.add_argument(
        "--rate",
        help="Adam optimizer Learning rate. [default: 0.01]",
        type=float,
        default=0.01,
    )
    group4.add_argument(
        "--amp",
        help="Amplitude of the GP kernel. [default: 1.0]",
        type=float,
        default=1.0,
    )
    group4.add_argument(
        "--lscale",
        help="The length scale of the GP kernel. [default: 1.0]",
        type=float,
        default=1.0,
    )
    group4.add_argument(
        "--maxiters",
        help="Maximum iterations for optimising Gaussian Process hyperparameters. "
        "For k-fold cross-validation, two inputs are required - one for "
        "training per fold and the other for training using train-test "
        "split. \nFor active learning and train-test split, a single input "
        "is required. [default: 0 i.e no GP training]",
        nargs="+",
        type=int,
        default=[0],
    )

    group5 = parser.add_argument_group("Active Learning Options")
    group5.add_argument(
        "--noactive",
        action="store_true",
        help="Do not perform active learning [default: False]",
        default=False,
    )
    group5.add_argument(
        "--samp",
        help="Type of sampling for active learning. Use random or entropy "
        "[default: entropy]",
        type=str,
        default="entropy",
        choices=["entropy", "random"],
    )
    group5.add_argument(
        "--cycle",
        help="Number of structures to sample and maximum number of times to "
        "sample separated by spaces for active learning. [default: 1 5]",
        nargs=2,
        type=int,
        default=[1, 5],
    )
    group5.add_argument(
        "--repeat",
        action="store_true",
        help="MEGNet train and pre-process activations in each active learning "
        "cycle [default: False]",
        default=False,
    )
    group5.add_argument(
        "--quan",
        help="Quantity of data for norepeat active learning [default: 1000]",
        type=int,
        default=1000,
    )
    group5.add_argument(
        "--stop",
        help="Minimum fraction of test set required for active learning [default: 0.1]",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()
    if args.nomeg:
        logging.info("No MEGNet training requested ...")
        sys.exit("No other network implemented!")
    else:
        if args.epoch > 0:
            logging.info("MEGNet training requested...")
            if args.prev:
                logging.info("Use a pre-trained MEGNet model in MEGNet training ...")
            else:
                logging.info(
                    "Do not use a pre-trained MEGNet model in MEGNet training ..."
                )

    # Check passed arguments before proceeding with the data processing
    _check_args(args)

    properties = load_data(args.data)
    for prop in properties:
        if args.noactive:
            if not args.nomeg:
                (
                    model,
                    activations_input_full,
                    Xfull,
                    yfull,
                    Xpool,
                    ypool,
                    Xtest,
                    ytest,
                ) = megnet_input(
                    prop,
                    args.include,
                    args.bond,
                    args.nfeat,
                    args.rcut,
                    args.width,
                    args.frac,
                )

            if args.nsplit == 1:
                # *****************************
                # TRAIN-TEST SPLIT APPROACH
                # *****************************
                datadir = "train_test_split/%s_results" % prop

                if not args.nomeg and args.epoch > 0:
                    logging.info("Training MEGNet on the pool ...")
                    training.train_test_split(
                        datadir,
                        prop,
                        args.prev,
                        model,
                        args.batch,
                        args.epoch,
                        Xpool,
                        ypool,
                        Xtest,
                        ytest,
                    )

                logging.info("Obtaining latent points for the full dataset ...")
                latent_pool, latent_test = latent.train_test_split(
                    datadir,
                    prop,
                    args.layer,
                    activations_input_full,
                    Xpool,
                    ytest,
                    args.perp,
                    args.ndims,
                    args.niters,
                )

                logging.info("Gaussian Process initiated ...")
                (
                    OptLoss,
                    OptAmp,
                    OptLength,
                    Optmae,
                    Optmse,
                    Optsae,
                    gp_mean,
                    gp_stddev,
                    R,
                ) = adam.train_test_split(
                    datadir,
                    prop,
                    latent_pool,
                    latent_test,
                    ypool,
                    ytest,
                    maxiters,
                    args.amp,
                    args.lscale,
                    args.rate,
                )

                logging.info(
                    "Saving optimised hyperparameters and GP posterior plots ..."
                )
                plot.train_test_split(
                    datadir,
                    prop,
                    args.layer,
                    maxiters,
                    args.rate,
                    OptLoss,
                    OptAmp,
                    OptLength,
                    ytest,
                    gp_mean,
                    gp_stddev,
                    None,
                    None,
                    Optmae,
                    Optmse,
                    Optsae,
                    R,
                )

            elif args.nsplit > 1:
                # ***************************
                # K-FOLD CROSS VALIDATION
                # ***************************

                OptAmp_fold = np.array([])
                OptLength_fold = np.array([])
                Optmae_val_fold = np.array([])
                Optmse_val_fold = np.array([])
                mae_test_fold = np.array([])
                kf = KFold(n_splits=args.nsplit, shuffle=True, random_state=0)
                for fold, (train_idx, val_idx) in enumerate(kf.split(Xpool)):
                    datadir = "k_fold/%s_results/0%s_fold" % (prop, fold)
                    Xtrain, Xval = Xpool[train_idx], Xpool[val_idx]
                    ytrain, yval = ypool[train_idx], ypool[val_idx]

                    if not args.nomeg and args.epoch > 0:
                        print("\nTraining MEGNet on fold %s training set ..." % fold)
                        training.k_fold(
                            datadir,
                            fold,
                            prop,
                            args.prev,
                            model,
                            args.batch,
                            args.epoch,
                            Xtrain,
                            ytrain,
                            Xval,
                            yval,
                        )

                    logging.info("Obtaining latent points for the full dataset ...")
                    latent_train, latent_val, latent_test = latent.k_fold(
                        datadir,
                        fold,
                        prop,
                        args.layer,
                        activations_input_full,
                        train_idx,
                        val_idx,
                        Xpool,
                        args.perp,
                        args.ndims,
                        args.niters,
                    )

                    logging.info("Gaussian Process initiated ...")
                    amp, args.lscale, Optmae_val, Optmse_val, mae_test = adam.k_fold(
                        datadir,
                        prop,
                        latent_train,
                        latent_val,
                        latent_test,
                        ytrain,
                        yval,
                        ytest,
                        maxiters[0],
                        args.amp,
                        args.lscale,
                        args.rate,
                    )
                    OptAmp_fold = np.append(OptAmp_fold, amp)
                    OptLength_fold = np.append(OptLength_fold, args.lscale)
                    Optmae_val_fold = np.append(Optmae_val_fold, Optmae_val)
                    Optmse_val_fold = np.append(Optmse_val_fold, Optmse_val)
                    mae_test_fold = np.append(mae_test_fold, mae_test)
                if all(Optmae_val_fold):
                    print(
                        "\nCross-validation statistics: MAE = %.4f, MSE = %.4f"
                        % (Optmae_val_fold.mean(), Optmse_val_fold.mean())
                    )
                logging.info("Cross-validation complete!")

                print("")
                # Choose the best fitted model for the train-test split training
                logging.info("Training MEGNet on the pool ...")
                if args.prev:
                    prev = "k_fold/%s_results/0%s_fold/model-best-new-%s.h5" % (
                        prop,
                        np.argmin(Optmae_val_fold),
                        prop,
                    )
                    print("The selected best fitted model: %s" % prev)
                    args.prev = prev
                datadir = "k_fold/%s_results" % prop
                if not args.nomeg and args.epoch > 0:
                    training.train_test_split(
                        datadir,
                        prop,
                        args.prev,
                        model,
                        args.batch,
                        args.epoch,
                        Xpool,
                        ypool,
                        Xtest,
                        ytest,
                    )

                logging.info("Obtaining latent points for the full dataset ...")
                latent_pool, latent_test = latent.train_test_split(
                    datadir,
                    prop,
                    args.layer,
                    activations_input_full,
                    Xpool,
                    ytest,
                    args.perp,
                    args.ndims,
                    args.niters,
                )

                logging.info("Gaussian Process initiated ...")
                (
                    OptLoss,
                    OptAmp,
                    OptLength,
                    Optmae,
                    Optmse,
                    Optsae,
                    gp_mean,
                    gp_stddev,
                    R,
                ) = adam.train_test_split(
                    datadir,
                    prop,
                    latent_pool,
                    latent_test,
                    ypool,
                    ytest,
                    maxiters[1],
                    args.amp,
                    args.lscale,
                    args.rate,
                )

                logging.info(
                    "Saving optimised hyperparameters and GP posterior plots ..."
                )
                plot.train_test_split(
                    datadir,
                    prop,
                    args.layer,
                    maxiters[1],
                    args.rate,
                    OptLoss,
                    OptAmp,
                    OptLength,
                    ytest,
                    gp_mean,
                    gp_stddev,
                    Optmae_val_fold,
                    mae_test_fold,
                    Optmae,
                    Optmse,
                    Optsae,
                    R,
                )
        else:
            # Perform active learning
            from aux.pool_sampling import ActiveLearning

            query, max_query = args.cycle[0], args.cycle[1]
            print("Number of cycle(s): ", max_query)
            print("Number of samples to move per cycle: ", query)

            if args.repeat:
                # ActiveLearning.repeat()

                # ********************************************
                # ACTIVE LEARNING WITH CYCLES OF NETWORK
                # TRAINING AND ACTIVATION EXTRACTION ANALYSIS
                # ********************************************
                training_data = np.array([])
                Optmae_val_cycle = np.array([])
                mae_test_cycle = np.array([])
                mse_test_cycle = np.array([])
                sae_test_cycle = np.array([])

                if not args.nomeg:
                    (
                        model,
                        activations_input_full,
                        Xfull,
                        yfull,
                        Xpool,
                        ypool,
                        Xtest,
                        ytest,
                        Xtrain,
                        ytrain,
                        Xval,
                        yval,
                    ) = megnet_input(
                        prop,
                        args.include,
                        args.bond,
                        args.nfeat,
                        args.rcut,
                        args.width,
                        args.frac,
                    )

                # Ensure there is adequate data in test set before proceeding
                assert (query * max_query) < int(args.stop * len(ytest)), (
                    "Test set size should be at least %s%% the dataset after active learning. Reduce stop and/or cycle parameters!"
                    % args.stop
                )

                for i in range(max_query + 1):
                    print("\nQuery number ", i)
                    datadir = "active_learn/repeat/%s_results/%s/0%s_model" % (
                        prop,
                        args.samp,
                        i,
                    )

                    if not args.nomeg and args.epoch > 0:
                        logging.info("Training MEGNet on the pool ...")
                        training.active(
                            datadir,
                            i,
                            prop,
                            args.prev,
                            model,
                            args.samp,
                            args.batch,
                            args.epoch,
                            Xpool,
                            ypool,
                            Xtest,
                            ytest,
                        )

                    logging.info("Obtaining latent points for the full dataset ...")
                    latent_train, latent_val, latent_test = latent.active(
                        datadir,
                        prop,
                        args.layer,
                        args.samp,
                        activations_input_full,
                        Xfull,
                        Xtest,
                        ytest,
                        Xtrain,
                        Xval,
                        args.perp,
                        args.ndims,
                        args.niters,
                    )

                    logging.info("Gaussian Process initiated ...")
                    (
                        OptLoss,
                        OptAmp,
                        OptLength,
                        args.amp,
                        args.lscale,
                        gp_mean,
                        gp_stddev,
                        gp_variance,
                        Optmae_val,
                        mae_test,
                        mse_test,
                        sae_test,
                        R,
                    ) = adam.active(
                        datadir,
                        prop,
                        latent_train,
                        latent_val,
                        latent_test,
                        ytrain,
                        yval,
                        ytest,
                        maxiters,
                        args.amp,
                        args.lscale,
                        args.rate,
                    )

                    # Save some parameters for plotting purposes.
                    training_data = np.append(training_data, len(ytrain))
                    Optmae_val_cycle = np.append(Optmae_val_cycle, Optmae_val)
                    mae_test_cycle = np.append(mae_test_cycle, mae_test)
                    mse_test_cycle = np.append(mse_test_cycle, mse_test)
                    sae_test_cycle = np.append(sae_test_cycle, sae_test)

                    logging.info(
                        "Saving optimised hyperparameters and GP posterior plots ..."
                    )
                    plot.active(
                        datadir,
                        prop,
                        args.layer,
                        maxiters,
                        args.rate,
                        OptLoss,
                        OptAmp,
                        OptLength,
                        args.samp,
                        query,
                        training_data,
                        ytest,
                        gp_mean,
                        gp_stddev,
                        Optmae_val_cycle,
                        mae_test_cycle,
                        mae_test,
                        mse_test,
                        sae_test,
                        R,
                    )

                    # Sample using variance on the predictions
                    if i < max_query:
                        if args.samp == "entropy":
                            if i == 0:
                                logging.info(
                                    "Entropy sampling for active learning enabled ..."
                                )
                            Xpool, ypool, Xtrain, ytrain, Xtest, ytest = (
                                EntropySelection(
                                    i,
                                    Xtrain,
                                    ytrain,
                                    Xtest,
                                    ytest,
                                    Xval,
                                    yval,
                                    gp_variance,
                                    query,
                                    max_query,
                                )
                            )
                        elif args.samp == "random":
                            if i == 0:
                                logging.info(
                                    "Random sampling for active learning enabled ..."
                                )
                            Xpool, ypool, Xtrain, ytrain, Xtest, ytest = (
                                RandomSelection(
                                    i,
                                    Xtrain,
                                    ytrain,
                                    Xtest,
                                    ytest,
                                    Xval,
                                    yval,
                                    gp_variance,
                                    query,
                                    max_query,
                                )
                            )
                    elif i == max_query:
                        if os.path.isdir("callback/"):
                            subprocess.call(["rm", "-r", "callback"])

            else:
                # ************************************
                # ACTIVE LEARNING WITHOUT CYCLES OF
                # NETWORK TRAINING AND tSNE ANALYSIS
                # *************************************
                val_frac = args.frac[0]
                training_data = np.array([])
                Optmae_val_cycle = np.array([])
                mae_test_cycle = np.array([])
                mse_test_cycle = np.array([])
                sae_test_cycle = np.array([])
                samp_idx = np.array([])

                if not args.nomeg:
                    model, activations_input_full, Xfull, yfull = megnet_input(
                        prop,
                        args.include,
                        args.bond,
                        args.nfeat,
                        args.rcut,
                        args.width,
                        args.frac,
                        args.quan,
                    )

                datadir = "active_learn/norepeat/%s_results/%s_model" % (
                    prop,
                    args.quan,
                )
                if not os.path.isdir(datadir):
                    os.makedirs(datadir)

                Xpool = Xfull[: args.quan]
                ypool = yfull[: args.quan]
                Xtest = Xfull[args.quan :]
                ytest = yfull[args.quan :]

                # Ensure there is adequate data in test set before proceeding
                assert (query * max_query) < int(args.stop * len(ytest)), (
                    "Test set size should be at least %s%% the dataset after active learning. Reduce stop and/or cycle parameters!"
                    % args.stop
                )

                val_boundary = int(len(Xpool) * val_frac)
                Xtrain = Xpool[:-val_boundary]
                ytrain = ypool[:-val_boundary]
                Xval = Xpool[-val_boundary:]
                yval = ypool[-val_boundary:]

                print("Requested validation set: %s%% of pool" % (val_frac * 100))
                print("Training set:", ytrain.shape)
                print("Validation set:", yval.shape)
                print("Test set:", ytest.shape)

                logging.info("Saving the data to file ...")
                np.save("%s/ytrain.npy" % datadir, ytrain)
                np.save("%s/yval.npy" % datadir, yval)

                print("\nProcessing %s samples ..." % args.quan)
                # MEGNet train and tSNE analyse or scale features once
                if not args.nomeg and args.epoch > 0:
                    training.train_test_split(
                        datadir,
                        prop,
                        args.prev,
                        model,
                        args.batch,
                        args.epoch,
                        Xpool,
                        ypool,
                        Xtest,
                        ytest,
                    )

                logging.info("Obtaining latent points for the full dataset ...")
                latent.active(
                    datadir,
                    prop,
                    args.layer,
                    args.samp,
                    activations_input_full,
                    Xfull,
                    Xtest,
                    ytest,
                    Xtrain,
                    Xval,
                    args.perp,
                    args.ndims,
                    args.niters,
                )

                logging.info("Loading the latent points ...")
                latent_train = np.load("%s/latent_train.npy" % datadir)
                latent_test = np.load("%s/latent_test.npy" % datadir)
                latent_val = np.load("%s/latent_val.npy" % datadir)

                # Lets create a new data directory and dump GP results into it
                datadir = datadir + "/" + args.samp + "/%s_samples" % query
                if not os.path.isdir(datadir):
                    os.makedirs(datadir)

                for i in range(max_query + 1):
                    print("\nQuery number ", i)

                    # Run the Gaussian Process
                    # GP train only at query 0 for the best hyperparameters
                    # required for the subsequent queries
                    if i == 0:
                        (
                            OptLoss,
                            OptAmp,
                            OptLength,
                            amp,
                            args.lscale,
                            gp_mean,
                            gp_stddev,
                            gp_variance,
                            Optmae_val,
                            mae_test,
                            mse_test,
                            sae_test,
                            R,
                        ) = adam.active(
                            datadir,
                            prop,
                            latent_train,
                            latent_val,
                            latent_test,
                            ytrain,
                            yval,
                            ytest,
                            maxiters,
                            args.amp,
                            args.lscale,
                            args.rate,
                        )
                    else:
                        maxiters = 0
                        (
                            OptLoss,
                            OptAmp,
                            OptLength,
                            Amp,
                            Length_Scale,
                            gp_mean,
                            gp_stddev,
                            gp_variance,
                            Optmae_val,
                            mae_test,
                            mse_test,
                            sae_test,
                            R,
                        ) = adam.active(
                            datadir,
                            prop,
                            latent_train,
                            latent_val,
                            latent_test,
                            ytrain,
                            yval,
                            ytest,
                            maxiters,
                            args.amp,
                            args.lscale,
                            args.rate,
                        )
                        # Set the new hyperparameters to those from query 0
                        Amp = amp
                        Length_Scale = args.lscale

                    # Dump some parameters to an array for plotting purposes.
                    training_data = np.append(training_data, len(ytrain))
                    mae_test_cycle = np.append(mae_test_cycle, mae_test)
                    mse_test_cycle = np.append(mse_test_cycle, mse_test)
                    sae_test_cycle = np.append(sae_test_cycle, sae_test)
                    if maxiters > 0:
                        Optmae_val_cycle = np.append(Optmae_val_cycle, Optmae_val)

                    if i < max_query:
                        if args.samp == "entropy":
                            if i == 0:
                                logging.info(
                                    "Entropy sampling for active learning enabled ..."
                                )
                            (
                                idx,
                                latent_pool,
                                ypool,
                                latent_train,
                                ytrain,
                                latent_test,
                                ytest,
                            ) = EntropySelection(
                                i,
                                latent_train,
                                ytrain,
                                latent_test,
                                ytest,
                                latent_val,
                                yval,
                                gp_variance,
                                query,
                                max_query,
                            )
                        elif args.samp == "random":
                            if i == 0:
                                logging.info(
                                    "Random sampling for active learning enabled ..."
                                )
                            (
                                idx,
                                latent_pool,
                                ypool,
                                latent_train,
                                ytrain,
                                latent_test,
                                ytest,
                            ) = RandomSelection(
                                i,
                                latent_train,
                                ytrain,
                                latent_test,
                                ytest,
                                latent_val,
                                yval,
                                gp_variance,
                                query,
                                max_query,
                            )
                        samp_idx = np.append(samp_idx, idx)

                logging.info("Writing the results to file ...")
                np.save("%s/training_data_for_plotting.npy" % datadir, training_data)
                np.save("%s/gp_mae.npy" % datadir, mae_test_cycle)
                np.save("%s/gp_mse.npy" % datadir, mse_test_cycle)
                np.save("%s/gp_sae.npy" % datadir, sae_test_cycle)
                np.save("%s/samp_indices.npy" % datadir, samp_idx)
                np.save("%s/Xtest.npy" % datadir, np.delete(Xtest, samp_idx))
                if maxiters > 0:
                    np.save("%s/val_mae.npy" % datadir, Optmae_val_cycle)

                logging.info("Saving plots ...")
                plot.norepeat(datadir, prop, args.layer, args.samp, query, maxiters)


if __name__ == "__main__":
    main()
