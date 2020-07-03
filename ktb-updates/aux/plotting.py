"""
gp-net.py, SciML-SCD, RAL 

For producing plots.
"""

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt



class plot:

    def active(prop, layer, rate, query, mae_val_entropy, mae_val_random,
               mae_gp_entropy, mae_gp_random):
        """
        plot.active(prop, layer, rate, query, mae_val_entropy, mae_val_random, 
                    mae_gp_entropy, mae_gp_random):

        Inputs:
        prop-                   Optical property of interest.  
        layer-                  Neural network layer of interest. 
        rate-                   Learning rate for Adam optimisation. 
        query-                  Number of samples to move from the 
                                test set into the pool. 
        mae_val_entropy-        MAE on validation set in entropy 
                                sampling. 
        mae_val_random-         MAE on validation set in random
                                sampling.
        mae_gp_entropy-         MAE on GP prediction in entropy
                                sampling.
        mae_gp_random-          MAE on GP prediction in random
                                sampling. 

        Outputs:
        1-                      A plot of the MAE for entropy
                                and random sampling vs total number 
                                of queries.
        """
        plt.figure(figsize = [14, 8])
        plt.suptitle("GP results of %s layer: Adam optimisation at learning rate %s \nSamples per query = %s"
                     %(layer, rate, query))
        plt.subplot(211)
        plt.plot(mae_val_entropy, "r", label="Entropy-based sampling")
        plt.plot(mae_val_random, "b", label="Random-based sampling")
        plt.xlabel("Active learning iterations")
        if prop == "band_gap":
            plt.ylabel("MAE on validation set [eV]")
        else:
            plt.ylabel("MAE on validation set [eV/atom]")
        plt.legend()

        plt.subplot(212)
        plt.plot(mae_gp_entropy, "r", label="Entropy-based sampling")
        plt.plot(mae_gp_random, "b", label="Random-based sampling")
        plt.xlabel("Active learning iterations")
        if prop == "band_gap":
            plt.ylabel("MAE on GP prediction [eV]")
        else:
            plt.ylabel("MAE on GP prediction [eV/atom]")
            plt.legend()
            
        plt.savefig("active_learn.pdf")
