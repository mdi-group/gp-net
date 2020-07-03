#!/anaconda3/bin/python

"""
gp-net.py, SciML-SCD, RAL

A tool for inserting uncertainties into a neural network. 
"""

import argparse  
import sys
sys.path.append('../../')
from aux.get_info import megnet_input
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
        # For active learning 
        self.noactive = False
        self.cycle = 20, 5
        
        # For MEGNet only 
        self.meg = False 
        self.cnn = False
        self.bond = 10
        self.g = 2
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
        self.l = "readout_0" # for MEGNet        
        self.perp = 50
        self.niters = 1000
        self.ndims = 2
        
        # GP specific arguments
        self.rate = 0.01 
        self.amp = 1.0
        self.length = 1.0
        self.opt = "adam"
        self.maxiters = 0,0

def main():
    """ From command line, all parsing are handled here """
    parser = argparse.ArgumentParser(description="Uncertainty quantification in neural networks.")
    parser.add_argument("-meg", action="store_true", help="Train with MEGNet. [default: False]",
                        default=False)
    parser.add_argument("-cnn", action="store_true", help="Train with CNN (Not implemented).",
                        default=False)
    parser.add_argument("-noactive", action="store_true",
                        help="Don't do active learning [default: False]", default=False)
    parser.add_argument("-cycle", help="Number of structures to sample and maximum number of times\
                        to sample separated by spaces for active learning. [default: 20 5]",
                        nargs=2, type=int)
    parser.add_argument("-samp", help="Type of sampling for active learning. Use random or\
                         entropy [No default]", type=str)
    parser.add_argument("-data", 
                        help="Input dataset(s). Multiple datasets can be passed, one\
                        per optical property of interest. [No default]", type=str, nargs="+")
    parser.add_argument("-key", 
                        help="API key for data download and the optical properties of\
                        interest, separated by spaces. For MEGNet users only. [eg. Key band_gap\
                        formation_energy_per_atom e_above_hull]", type=str, nargs="+")
    parser.add_argument("-frac", 
                        help="Fraction of data for training and fraction of the training set\
                        for validation separated by spaces. [default: 0.3 0.7]", nargs=2,
                        type=float)
    parser.add_argument("-nsplit",
                        help="Number of training set splits for k-fold cross-validation.\
                        [default: 1 i.e no cross-validation]", type=int)    
    parser.add_argument("-bond", help="MEGNet feature bond. [default: 10]", type=int)
    parser.add_argument("-g", help="MEGNet feature global. [default: 2]", type=int) 
    parser.add_argument("-c", "--cutoff", help="MEGNet radial cutoff. [default: 5]", type=int)
    parser.add_argument("-w", "--width", help="MEGNet gaussian width. [default: 0.5]", type=float)
    parser.add_argument("-include", action="store_true",
                        help="Include zero optical property values in the MEGNet training\
                        and/or Gaussian process analysis. [default: False]", default=False)
    parser.add_argument("-epochs", 
                        help="Epochs. [default: 0 ie. Perform no training with MEGNet or CNN]",
                        type=int)
    parser.add_argument("-batch",
                        help="Batch size for training with MEGNet or CNN. [default: 256]",
                        type=int)
    parser.add_argument("-prev", action="store_true",
                       help="Use a pre-trained MEGNet model during training with MEGNet.\
                       [default: False]", default=False)
    parser.add_argument("-l",
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
                        help="Dimensions of embedded space in tSNE. [default: 2]", type=int)
    parser.add_argument("-opt", 
                        help="Optimizer for optimizing GP hyperparameters. [default: adam]",
                        type=str)
    parser.add_argument("-rate", 
                        help="Adam optimizer Learning rate. [default: 0.01]", type=float)
    parser.add_argument("-amp", 
                        help="Amplitude of the GP kernel. [default: 1]", type=float)
    parser.add_argument("-length",
                        help="The length scale of the GP kernel. [default: 1]", type=float)
    parser.add_argument("-maxiters",
                        help="Maximum iterations for optimising GP hyperparameters. For\
                        k-fold cross-validation, maxiters for training per fold and\
                        maxiters for training the train-test split.\nIf no k-fold\
                        cross-validation, a single input is required\
                        [default: 0 0 i.e no MEGNet/CNN and GP training]", nargs="+",
                        type=int)

    args = parser.parse_args()
    cycle = args.cycle or Params().cycle 
    fraction = args.frac or Params().frac
    nsplit = args.nsplit or Params().nsplit
    ZeroVals = args.include or Params().include     
    bond = args.bond or Params().bond
    nfeat_global = args.g or Params().g
    cutoff = args.cutoff or Params().cutoff
    width = args.width or Params().width
    epochs = args.epochs or Params().epochs
    batch = args.batch or Params().batch
    prev = args.prev or Params().prev
    layer = args.l or Params().l
    perp = args.perp or Params().perp
    niters = args.niters or Params().niters
    ndims = args.ndims or Params().ndims
    rate = args.rate or Params().rate 
    amp = args.amp or Params().amp
    length_scale = args.length or Params().length
    opt = args.opt or Params().opt
    maxiters = args.maxiters or Params().maxiters

    # Display layers in a pre-fitted MEGNet model 
    if args.ltype:
        from aux.get_info import show_layers
        show_layers(args.ltype)
        sys.exit()

    # Train with MEGNet or CNN  
    if (args.meg or args.cnn) == False:
        sys.exit("Set -meg or -cnn!")

    if args.data or (args.data and args.key):
        from aux.get_info import load_data
        properties = load_data(args.data)
    elif args.key:
        from aux.get_info import download
        properties = download(args.key)
    else:
        sys.exit("No input data provided. Use -data or -key option!")

    for prop in properties:
        if args.meg:
            print("MEGNet training requested ...")
            (model, activations_input_full, Xfull, yfull, Xpool, ypool,
             Xtest, ytest, Xtrain, ytrain, Xval, yval)  = megnet_input(prop, ZeroVals, bond, nfeat_global,
                                                                       cutoff, width, fraction)
            
        if not args.noactive: 
            print("\nActive learning requested ...")
            import subprocess
            from train.MEGNetTrain import training 
            from aux.activations import latent
            from aux.pool_sampling import selection_fn

            if fraction[1] >= 1:
                sys.exit("The second parameter to -frac must be less than 1!")
            if len(maxiters) > 1:
                sys.exit("-maxiters must have length 1!")
            else:
                maxiters = maxiters[0]

            query = cycle[0]
            max_query = cycle[1]
            training_data = [ ]
            mae_val_entropy = [ ]
            mae_gp_entropy = [ ]           
            mae_val_random = [ ]
            mae_gp_random = [ ]
            print("Number of cycle(s): ", max_query)
            print("Number of samples per cycle: ", query)
            i = 0
            while i < max_query + 1:
                print("\nQuery number ", i)
                if args.meg:
                    print("Training MEGNet on the pool ...")                    
                    if i == 0:
                        training.active(i, prop, prev, model, args.samp, batch, 2,
                                    Xpool, ypool, Xtest, ytest)
                        training_data.append(len(ytrain))
                    else:
                        training.active(i, prop, prev, model, args.samp, batch, epochs,
                                    Xpool, ypool, Xtest, ytest)
                        training_data.append(len(ytrain))
                    
                print("\nObtaining latent points for the full dataset ...")
                tsne_train, tsne_val, tsne_test = latent.active(i, prop, perp, layer, args.samp,
                                                                activations_input_full, Xfull, Xpool,
                                                                Xtest,  Xtrain, Xval, ndims, niters)

                print("\nGP Training on the DFT %s ..." %prop)
                ampo = amp
                leno = length_scale
                gprm_dft, dft_variance, mae_val, mae_gp, amp, length_scale = adam.active(
                    tsne_train, tsne_val, tsne_test, ytrain, yval, ytest, maxiters, amp, length_scale, rate)
                amp = ampo
                length_scale = leno

                if args.samp == "entropy":
                    mae_val_entropy.append(mae_val)
                    mae_gp_entropy.append(mae_gp)
                    EntropySelection = selection_fn.EntropySelection                    
                    Xpool, ypool, Xtrain, ytrain, Xtest, ytest = EntropySelection(i, Xtrain, ytrain, Xtest, ytest,
                                                                                  Xval, yval, dft_variance, query,
                                                                                  max_query)
                elif args.samp == "random":
                    mae_val_random.append(mae_val)
                    mae_gp_random.append(mae_gp)
                    RandomSelection = selection_fn.RandomSelection                    
                    Xpool, ypool, Xtrain, ytrain, Xtest, ytest = RandomSelection(i, Xtrain, ytrain, Xtest, ytest, Xval,
                                                                                 yval, dft_variance, query, max_query)
                else:
                    sys.exit("Sampling type not recognised")
                if i == max_query:
                    subprocess.call(["rm", "-r", "callback"])
                i += 1

            if maxiters > 0:
                print("\nSaving active learning plots")
                from aux.plotting import plot

                if args.samp == "entropy":
                    plot.active(prop, layer, rate, query, args.samp, mae_val_entropy, 
                                mae_gp_entropy, training_data) 
                elif args.samp == "random":
                    plot.active(prop, layer, rate, query, args.samp, mae_val_random, 
                                mae_gp_random, training_data) 
                







                    
                

if __name__ == "__main__":
    print ("\ngp-net.py ver ", VERSION)
    main()
