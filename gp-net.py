#!/anaconda3/bin/python

"""
gp-net.py, SciML-SCD, RAL

A tool for inserting uncertainties into a neural network. 
"""
import argparse  
import sys
import logging
import os 
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(levelname)s:gp-net: %(message)s")
import numpy as np

from aux.get_info import megnet_input
from aux.activations import latent
from aux.plotting import plot
from train.MEGNetTrain import training
from optimizers.adam import adam 

VERSION = "1.0"

class Params:

    def __init__(self):
        """
        Params()
        
        Initialises the parameter list with default values. 
        These parameters can be changed by accessing these 
        class variables.
        """
        # For checking data
        self.checkdata = False
        
        # Specific to active learning 
        self.noactive = False
        self.norepeat = False 
        self.samp = "entropy"
        self.cycle = 20, 5
        
        # For MEGNet only 
        self.nomeg = False 
        self.bond = 10
        self.nfeat = 2
        self.cutoff = 5
        self.width = 0.5
        self.include = False
        self.batch = 256
        self.prev = False
        
        # For both MEGNet and GP
        self.epochs = 0
        self.frac = 0.3, 0.7 
        self.nsplit = 1
        
        # For tSNE only 
        self.layer = "readout_0" # for MEGNet        
        self.perp = 50
        self.niters = 1000
        self.ndims = 0
        
        # GP specific arguments
        self.rate = 0.01 
        self.amp = 10.0
        self.length = 10.0        
        self.maxiters = 0,0


def main():
    """ From command line, all parsing are handled here """
    parser = argparse.ArgumentParser(description="Uncertainty quantification in neural networks.")
    parser.add_argument("-nomeg", action="store_true",
                        help="Do not train with MEGNet. [default: False]", default=False)
    parser.add_argument("-noactive", action="store_true",
                        help="Don't do active learning [default: False]", default=False)
    parser.add_argument("-samp", help="Type of sampling for active learning. Use random or\
                         entropy [default: entropy]", type=str)
    parser.add_argument("-cycle", help="Number of structures to sample and maximum number of times\
                        to sample separated by spaces for active learning. [default: 20 5]",
                        nargs=2, type=int)
    parser.add_argument("-norepeat", action="store_true",
                        help="Do not train with MEGNet in each active learning cycle [default: False]",
                        default=False)
    parser.add_argument("-q", "--quan", help="Quantity of data for norepeat active learning [No default]",
                        type=int)
    parser.add_argument("-data", 
                        help="Input dataset(s). Multiple datasets can be passed, one\
                        per optical property of interest. [No default]", type=str, nargs="+")
    parser.add_argument("-checkdata", action="store_true",
                        help="Check number of entries in the dataset. [default: False]",
                        default = False)
    parser.add_argument("-key", 
                        help="API key for data download and the optical properties of\
                        interest, separated by spaces. For MEGNet users only. [eg. Key band_gap\
                        formation_energy_per_atom e_above_hull]", type=str, nargs="+")
    parser.add_argument("-frac", 
                        help="Fraction of data for training and fraction of the training set\
                        for validation separated by spaces. [default: 0.3 0.7]", nargs=2,
                        type=float)
    parser.add_argument("-include", action="store_true",
                        help="Include zero optical property values in the MEGNet training\
                        and/or Gaussian process analysis. [default: False]", default=False)
    parser.add_argument("-nsplit",
                        help="Number of training set splits for k-fold cross-validation.\
                        [default: 1 i.e no cross-validation]", type=int)    
    parser.add_argument("-epochs", 
                        help="Epochs. [default: 0 ie. Perform no training with MEGNet]",
                        type=int)
    parser.add_argument("-batch",
                        help="Batch size for training with MEGNet or CNN. [default: 256]",
                        type=int)
    parser.add_argument("-bond", help="MEGNet feature bond. [default: 10]", type=int)
    parser.add_argument("-nfeat", help="MEGNet feature global. [default: 2]", type=int) 
    parser.add_argument("-cutoff", "--cutoff", help="MEGNet radial cutoff. [default: 5]",
                        type=int)
    parser.add_argument("-width", "--width", help="MEGNet gaussian width. [default: 0.5]",
                        type=float)    
    parser.add_argument("-prev", action="store_true",
                       help="Use a pre-trained MEGNet model during training with MEGNet.\
                       [default: False]", default=False)
    parser.add_argument("-layer",
                        help="MEGNet fitted model layer to analyse. [default: readout_0 i.e 32\
                        dense layer]", type=str)
    parser.add_argument("-ltype", help="Display the layers in a fitted MEGNet model.",
                        type=str)
    parser.add_argument("-p", "--perp", 
                        help="Perplexity value to use in dimension reduction with tSNE.\
                        [default: 50]", type=float)
    parser.add_argument("-niters",
                        help="Number of iterations for optimisation in tSNE. [default: 1000]",
                        type=int)
    parser.add_argument("-ndims", 
                        help="Dimensions of embedded space. 0 => scale activations in 0,1 range\
                        2 or 3 => Reduce dimensions of activations with tSNE. [default: 0]",
                        type=int)
    parser.add_argument("-rate", 
                        help="Adam optimizer Learning rate. [default: 0.01]", type=float)
    parser.add_argument("-amp", 
                        help="Amplitude of the GP kernel. [default: 10.0]", type=float)
    parser.add_argument("-length",
                        help="The length scale of the GP kernel. [default: 10.0]", type=float)
    parser.add_argument("-maxiters",
                        help="Maximum iterations for optimising GP hyperparameters. For\
                        k-fold cross-validation, two inputs are required - one for training\
                        per fold and the other for training using train-test split.\
                        \nFor active learning and no k-fold cross-validation, a single input\
                        is required. [default: 0 0 i.e no MEGNet and GP training]",
                        nargs="+", type=int)
    
    args = parser.parse_args()
    samp = args.samp or Params().samp
    cycle = args.cycle or Params().cycle 
    fraction = args.frac or Params().frac
    nsplit = args.nsplit or Params().nsplit
    epochs = args.epochs or Params().epochs
    batch = args.batch or Params().batch
    bond = args.bond or Params().bond
    nfeat_global = args.nfeat or Params().nfeat
    cutoff = args.cutoff or Params().cutoff
    width = args.width or Params().width
    layer = args.layer or Params().layer
    perp = args.perp or Params().perp
    niters = args.niters or Params().niters
    ndims = args.ndims or Params().ndims
    rate = args.rate or Params().rate 
    amp = args.amp or Params().amp
    length_scale = args.length or Params().length
    maxiters = args.maxiters or Params().maxiters


    # Display layers in a pre-fitted MEGNet model 
    if args.ltype:
        from aux.get_info import show_layers
        show_layers(args.ltype)
        sys.exit()

    # Check number of entries in dataset 
    if args.checkdata:
        from aux.get_info import ReadData
        for prop in properties:
            ReadData(prop, args.include)
        sys.exit()    

    if args.nomeg: 
        logging.info("No MEGNet training requested ...")
        sys.exit("No other network implemented!")
    else:  
        logging.info("MEGNet training requested...")

    if args.prev:
        logging.info("Use a pre-trained MEGNet model in MEGNet training ...")
    else:
        logging.info("Do not use a pre-trained MEGNet model in MEGNet training ...")

    if args.include:
        logging.info("Include zero optical property values ...")
    else:
        logging.info("Exclude zero optical property values ...")

    # NB: fraction[0] is the pool fraction
    #     fraction[1] is the validation fraction 
    if args.noactive:
        logging.info("No active learning requested ...")
        if fraction[1] < 1 or fraction[1] > 1:
            logging.error("The second parameter to -frac must be equal to 1!")
            sys.exit()
        if nsplit == 1:
            logging.info("Train-test split approach requested ...")
            if len(maxiters) > 1:
                logging.error("-maxiters must have length 1!")
                sys.exit()
            else:
                maxiters = maxiters[0]
        else:
            print("%s-fold cross-validation requested ..." %nsplit)
            if len(maxiters) == 1 or len(maxiters) > 2:
                logging.error("-maxiters must have length 2!")
                sys.exit()            
    else:
        logging.info("Active learning to be performed ...")
        if samp == "entropy":
            logging.info("Entropy sampling for active learning enabled ...")
        elif samp == "random":
            logging.info("Random sampling for active learning enabled ...")
        if not args.norepeat:
            logging.info("MEGNet train and tSNE analyse per cycle of active learning ...")
        else:
            logging.info("MEGNet train and tSNE analyse ONCE during the active learning ...")            
            if not args.quan:
                logging.error("Provide quantity of data to use with -q or --quan!")
                sys.exit()
        if fraction[1] >= 1:
            logging.error("The second parameter to -frac must be less than 1!")
            sys.exit()
        if nsplit > 1:
            logging.error("Active learning with k-fold cross validation not supported!")
            sys.exit()
        if len(maxiters) > 1:
            logging.error("-maxiters must have length 1!")
            sys.exit()
        else:
            maxiters = maxiters[0]
            
    # Get data for processing 
    if args.data or (args.data and args.key):
        from aux.get_info import load_data
        properties = load_data(args.data)
    elif args.key:
        from aux.get_info import download
        properties = download(args.key)
    else:
        logging.error("No input data provided. Use -data or -key option!")
        sys.exit()

    for prop in properties:
        if args.noactive:
            if not args.nomeg:
                (model, activations_input_full, Xfull, yfull, Xpool,
                 ypool, Xtest, ytest, Xtrain, ytrain, Xval, yval)  = megnet_input(
                     prop, args.include, bond, nfeat_global, cutoff, width, fraction)
            if nsplit == 1:
                #*****************************
                # TRAIN-TEST SPLIT APPROACH 
                #*****************************
                datadir = "train_test_split/%s_results" %prop
                if not args.nomeg and epochs > 0:
                    logging.info("Training MEGNet on the pool ...")
                    training.train_test_split(datadir, prop, args.prev, model, batch,
                                              epochs, Xpool, ypool, Xtest, ytest)
                    
                logging.info("Obtaining latent points for the full dataset ...")
                tsne_pool, tsne_test = latent.train_test_split(
                    datadir, prop, layer, activations_input_full, Xpool, ytest, perp,
                    ndims, niters)
            
                logging.info("Gaussian Process initiated ...")
                OptLoss, OptAmp, OptLength, Optmae, Optmse, Optsae, gp_mean, gp_stddev, R =\
                    adam.train_test_split(datadir, prop, tsne_pool, tsne_test, ypool, ytest,
                                          maxiters, amp, length_scale, rate)

                logging.info("Saving optimised hyperparameters and GP posterior plots ...")
                plot.train_test_split(datadir, prop, layer, maxiters, rate, OptLoss, OptAmp,
                                      OptLength, ytest, gp_mean, gp_stddev, None, None, Optmae,
                                      Optmse, Optsae, R)
            elif nsplit > 1:
                #***************************
                # K-FOLD CROSS VALIDATION
                #***************************
                from sklearn.model_selection import KFold

                OptAmp_fold = np.array([])
                OptLength_fold = np.array([])
                Optmae_val_fold = np.array([])
                Optmse_val_fold = np.array([])
                mae_test_fold = np.array([])
                kf = KFold(n_splits=nsplit, shuffle=True, random_state=0)
                for fold, (train_idx, val_idx) in enumerate(kf.split(Xpool)):
                    datadir = "k_fold/%s_results/0%s_fold" %(prop, fold)
                    Xtrain, Xval = Xpool[train_idx], Xpool[val_idx]
                    ytrain, yval = ypool[train_idx], ypool[val_idx]

                    if not args.nomeg and epochs > 0:
                        print("\nTraining MEGNet on fold %s training set ..." %fold)
                        training.k_fold(datadir, fold, prop, args.prev, model, batch, epochs,
                                        Xtrain, ytrain, Xval, yval)

                    logging.info("Obtaining latent points for the full dataset ...")
                    tsne_train, tsne_val, tsne_test = latent.k_fold(
                        datadir, fold, prop, layer, activations_input_full, train_idx, val_idx,
                        Xpool, perp, ndims, niters)

                    logging.info("Gaussian Process initiated ...")
                    amp, length_scale, Optmae_val, Optmse_val, mae_test = adam.k_fold(
                        datadir, prop, tsne_train, tsne_val, tsne_test, ytrain, yval, ytest,
                        maxiters[0], amp, length_scale, rate)
                    OptAmp_fold = np.append(OptAmp_fold, amp) 
                    OptLength_fold = np.append(OptLength_fold, length_scale)
                    Optmae_val_fold = np.append(Optmae_val_fold, Optmae_val)
                    Optmse_val_fold = np.append(Optmse_val_fold, Optmse_val)
                    mae_test_fold = np.append(mae_test_fold, mae_test)
                if all(Optmae_val_fold): 
                    print("\nCross-validation statistics: MAE = %.4f, MSE = %.4f" %(
                        Optmae_val_fold.mean(), Optmse_val_fold.mean()))
                logging.info("Cross-validation complete!")

                print("")
                # Choose the best fitted model for the train-test split training             
                logging.info("Training MEGNet on the pool ...")
                if args.prev: 
                    prev = "k_fold/%s_results/0%s_fold/model-best-new-%s.h5" %(
                        prop, np.argmin(Optmae_val_fold), prop)
                    print("The selected best fitted model: %s" %prev)
                    args.prev = prev 
                datadir = "k_fold/%s_results" %prop
                if not args.nomeg and epochs > 0:
                    training.train_test_split(datadir, prop, args.prev, model, batch, epochs,
                                              Xpool, ypool, Xtest, ytest)
                    
                logging.info("Obtaining latent points for the full dataset ...")
                tsne_pool, tsne_test = latent.train_test_split(
                    datadir, prop, layer, activations_input_full, Xpool, ytest, perp, ndims, niters)
                
                logging.info("Gaussian Process initiated ...")
                OptLoss, OptAmp, OptLength, Optmae, Optmse, Optsae, gp_mean, gp_stddev, R =\
                    adam.train_test_split(datadir, prop, tsne_pool, tsne_test, ypool, ytest,
                                          maxiters[1], amp, length_scale, rate)

                logging.info("Saving optimised hyperparameters and GP posterior plots ...")
                plot.train_test_split(datadir, prop, layer, maxiters[1], rate, OptLoss, OptAmp,
                                      OptLength, ytest, gp_mean, gp_stddev, Optmae_val_fold,
                                      mae_test_fold, Optmae, Optmse, Optsae, R)
        else:
             import subprocess
             from aux.pool_sampling import selection_fn
             EntropySelection = selection_fn.EntropySelection
             RandomSelection = selection_fn.RandomSelection

             query = cycle[0]
             max_query = cycle[1]
             print("Number of cycle(s): ", max_query)
             print("Number of samples per cycle: ", query)

             if not args.norepeat:
                 #********************************************
                 # ACTIVE LEARNING WITH CYCLES OF NETWORK 
                 # TRAINING AND tSNE ANALYSIS
                 #********************************************
                 training_data = np.array([])
                 Optmae_val_cycle = np.array([])
                 mae_test_cycle = np.array([])
                 mse_test_cycle = np.array([])
                 sae_test_cycle = np.array([])             
                 
                 if not args.nomeg:
                     (model, activations_input_full, Xfull, yfull, Xpool, ypool, Xtest,
                      ytest, Xtrain, ytrain, Xval, yval) = megnet_input(
                          prop, args.include, bond, nfeat_global, cutoff, width, fraction)
                 i = 0
                 while i < max_query + 1:
                     print("\nQuery number ", i)
                     datadir = "%s/%s_results/0%s_model" %(samp, prop, i)
                     
                     if not args.nomeg and epochs > 0:
                         logging.info("Training MEGNet on the pool ...")
                         training.active(datadir, i, prop, args.prev, model, args.samp,
                                         batch, epochs, Xpool, ypool, Xtest, ytest)

                     logging.info("Obtaining latent points for the full dataset ...")
                     tsne_train, tsne_val, tsne_test = latent.active(
                         datadir, prop, layer, samp, activations_input_full, Xfull, Xtest,
                         ytest, Xtrain, Xval, perp, ndims, niters)
                     
                     logging.info("Gaussian Process initiated ...")
                     (OptLoss, OptAmp, OptLength, amp, length_scale, gp_mean, gp_stddev,
                      gp_variance, Optmae_val, mae_test, mse_test, sae_test, R) =\
                          adam.active(datadir, prop, tsne_train, tsne_val, tsne_test, ytrain,
                                      yval, ytest, maxiters, amp, length_scale, rate)
                     
                     # Dump some parameters to an array for plotting purposes.
                     training_data = np.append(training_data, len(ytrain))
                     Optmae_val_cycle = np.append(Optmae_val_cycle, Optmae_val)
                     mae_test_cycle = np.append(mae_test_cycle, mae_test) 
                     mse_test_cycle = np.append(mse_test_cycle, mse_test)
                     sae_test_cycle = np.append(sae_test_cycle, sae_test)
                     
                     logging.info("Saving optimised hyperparameters and GP posterior plots ...")
                     plot.active(datadir, prop, layer, maxiters, rate, OptLoss, OptAmp, OptLength,
                                 samp, query, training_data, ytest, gp_mean, gp_stddev,
                                 Optmae_val_cycle, mae_test_cycle, mae_test, mse_test, sae_test, R)
                     
                     if i != max_query:
                         if samp == "entropy":
                             Xpool, ypool, Xtrain, ytrain, Xtest, ytest = EntropySelection(
                                 i, Xtrain, ytrain, Xtest, ytest, Xval, yval, gp_variance, query, max_query)
                         elif samp == "random":
                             Xpool, ypool, Xtrain, ytrain, Xtest, ytest = RandomSelection(
                                 i, Xtrain, ytrain, Xtest, ytest, Xval, yval, gp_variance, query, max_query)
                         else:
                             logging.error("Sampling type not recognised!")
                             sys.exit()
                     else:
                         if os.path.isdir("callback/"):
                             subprocess.call(["rm", "-r", "callback"])                         
                     i += 1
             else:
                 import matplotlib
                 matplotlib.use("agg")
                 import matplotlib.pyplot as plt
                 #************************************
                 # ACTIVE LEARNING WITHOUT CYCLES OF
                 # NETWORK TRAINING AND tSNE ANALYSIS
                 #*************************************
                 pool_frac = fraction[0]
                 val_frac = fraction[1]
                 training_data = np.array([]) 
                 Optmae_val_cycle = np.array([])
                 mae_test_cycle = np.array([]) 
                 mse_test_cycle = np.array([]) 
                 sae_test_cycle = np.array([]) 
                 
                 if not args.nomeg:
                     model, activations_input_full, Xfull, yfull = megnet_input(
                         prop, args.include, bond, nfeat_global, cutoff, width,
                         fraction, args.quan)

                 print("Requested validation set: %s%% of pool" %(val_frac*100))
                 datadir = "no_cycle/%s_results/%s_model" %(prop, args.quan)
                 if not os.path.isdir(datadir):
                     os.makedirs(datadir)
                         
                 Xpool = Xfull[:args.quan]
                 ypool = yfull[:args.quan]
                 Xtest = Xfull[args.quan:]
                 ytest = yfull[args.quan:]

                 val_boundary = int(len(Xpool) * val_frac)
                 Xtrain = Xpool[:-val_boundary]
                 ytrain = ypool[:-val_boundary]
                 Xval = Xpool[-val_boundary:]
                 yval = ypool[-val_boundary:]
                 print("Training set:", ytrain.shape)
                 print("Validation set:", yval.shape)

                 logging.info("Saving the data to file ...")
                 np.save("%s/ytrain.npy" %datadir, arr=ytrain)
                 np.save("%s/yval.npy" %datadir, arr=yval)

                 print("\nProcessing %s samples ..." %args.quan)
                 # MEGNet train and tSNE analyse or scale features once 
                 if not args.nomeg and epochs > 0:
                     training.train_test_split(datadir, prop, args.prev, model, batch,
                                               epochs, Xpool, ypool, Xtest, ytest)
                     
                 logging.info("Obtaining latent points for the full dataset ...")
                 latent.active(datadir, prop, layer, samp, activations_input_full,
                               Xfull, Xtest, ytest, Xtrain, Xval, perp, ndims, niters)
                     
                 logging.info("Loading the latent points ...")
                 tsne_train = np.load("%s/tsne_train.npy" %datadir)
                 tsne_test = np.load("%s/tsne_test.npy" %datadir)
                 tsne_val = np.load("%s/tsne_val.npy" %datadir)

                 # Lets create a new data directory and dump GP results into it 
                 datadir = datadir + "/" + samp + "/%s_samples" %query
                 if not os.path.isdir(datadir):
                     os.makedirs(datadir)
                         
                 for i in range(max_query):
                     print("\nQuery number ", i)

                     # Run the Gaussian Process
                     # GP train only at query 0 for the best hyperparameters
                     # required for the subsequent queries 
                     if i == 0:
                         (OptLoss, OptAmp, OptLength, amp, length_scale, gp_mean, gp_stddev,
                          gp_variance, Optmae_val, mae_test, mse_test, sae_test, R) =\
                              adam.active(datadir, prop, tsne_train, tsne_val, tsne_test,
                                          ytrain, yval, ytest, maxiters, amp, length_scale, rate)
                     else: 
                         maxiters = 0
                         (OptLoss, OptAmp, OptLength, Amp, Length_Scale, gp_mean, gp_stddev,
                          gp_variance, Optmae_val, mae_test, mse_test, sae_test, R) =\
                              adam.active(datadir, prop, tsne_train, tsne_val, tsne_test,
                                          ytrain, yval, ytest, maxiters, amp, length_scale, rate)
                         # Set the new hyperparameters to those from query 0 
                         Amp = amp
                         Length_Scale = length_scale 

                     # Dump some parameters to an array for plotting purposes.
                     training_data = np.append(training_data, len(ytrain)) 
                     mae_test_cycle = np.append(mae_test_cycle, mae_test) 
                     mse_test_cycle = np.append(mse_test_cycle, mse_test) 
                     sae_test_cycle = np.append(sae_test_cycle, sae_test) 
                     if maxiters > 0:
                         Optmae_val_cycle = np.append(Optmae_val_cycle, Optmae_val) 

                     if i != max_query - 1:
                         if samp == "entropy":
                             tsne_pool, ypool, tsne_train, ytrain, tsne_test, ytest  =\
                                 EntropySelection(i, tsne_train, ytrain, tsne_test, ytest,
                                                  tsne_val, yval, gp_variance, query, max_query)
                         elif samp == "random":
                             tsne_pool, ypool, tsne_train, ytrain, tsne_test, ytest  =\
                                 RandomSelection(i, tsne_train, ytrain, tsne_test, ytest,
                                                 tsne_val, yval, gp_variance, query, max_query)
                                 
                 logging.info("Writing the results to file ...")
                 np.save("%s/training_data_for_plotting.npy" %datadir,
                         arr=training_data)
                 np.save("%s/gp_mae.npy" %datadir, arr=mae_test_cycle)
                 np.save("%s/gp_mse.npy" %datadir, arr=mse_test_cycle)
                 np.save("%s/gp_sae.npy" %datadir, arr=sae_test_cycle)
                 if maxiters > 0:
                     np.save("%s/val_mae.npy" %datadir, arr=Optmae_val_cycle)

                 logging.info("Saving plots ...")
                 plt.figure(figsize = [12, 6])
                 plt.plot(training_data, mae_test_cycle, marker="o", color="tab:red")
                 plt.xlabel("Number of training data") 
                 if prop == "band_gap":
                     plt.ylabel("MAE on GP prediction [eV]") 
                 else:
                     plt.ylabel("MAE on GP prediction [eV/atom]")
                 if maxiters > 0:
                     plt.title("Type of sampling: %s \nAdam optimisation at learning rate = %s \nSamples per cycle = %s" %
                               (samp, rate, query))
                 else:
                     plt.title("Type of sampling: %s \nSamples per cycle = %s" %(samp, query))
                         
                 plt.savefig("%s/active_learn_%s.pdf" %(datadir, prop))




if __name__ == "__main__":
    print ("\ngp-net.py ver ", VERSION)
    main()
