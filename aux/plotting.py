"""
gp-net.py, SciML-SCD, RAL 

For producing plots.
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


class plot:

    def train_test_split(datadir, prop, layer, maxiters, rate, OptLoss, OptAmp,
                         OptLength, ytest_dft, gp_mean, gp_stddev, best_mae_fold,
                         mae_test_fold, Optmae, Optmse, Optsae, R):
        """
        Produces uncertainty figures for the train test split approach.

        Inputs:
        datadir-          Directory into which results are written into.
        prop-             Optical property of interest.
        layer-            Neural network layer of interest. 
        maxiters-         Number of iterations for optimising hyperparameters.
        rate-             Learning rate for Adam optimisation.
        OptLoss-          Loss minimised during GP training.
        OptAmp-           Optimised kernel amplitude.
        OptLength-        Optimised kernel scale length. 
        ytest_dft-        DFT-calculated set for testing. 
        gp_mean-          GP predicted optical property values.
        gp_stddev-        Standard deviation on GP prediction.
        best_mae_fold-    Best MAEs on the validation set in k-fold 
                          crossvalidation or MAEs on the validation set from 
                          active learning. 
        mae_test_fold-    The MAE on the test set using the optimised 
                          hyperparameters obatined from the k-fold cross validation. 
        Optmae-           Mean absolute error.
        Optmse-           Mean squared error.
        Optsae-           Standard deviation on the absolute error. 
        R-                Pearson correlation coefficient.

        Output:
        1-                Figure showing the error in prediction 
                          vs the uncertainty in the prediction. 
        """
        plt.figure(figsize=[30, 20])
        if maxiters > 0:
            plt.suptitle("GP results of %s layer: Adam optimisation at learning rate %s"
                         %(layer, rate))
            plt.subplot(421)
            plt.plot(OptLoss, "r", label="DFT")
            plt.xlabel("Iterations")
            plt.ylabel("%s Loss" %prop)
            plt.legend()

            plt.subplot(422)
            plt.plot(OptAmp, OptLoss, "b", label="DFT")
            plt.xlabel("Amplitude of the GP kernel")
            plt.ylabel("%s Loss" %prop)
            plt.legend()
            
            plt.subplot(423)
            plt.plot(OptLength, OptLoss, "g", label="DFT")
            plt.xlabel("Width of the GP kernel")
            plt.ylabel("%s Loss" %prop)
            plt.legend()
            try:
                # Add subplot for k-fold cross validation 
                ax1 = plt.subplot(424)
                ax1.plot(best_mae_fold, marker="o", color="tab:red")
                ax1.tick_params(axis="y", labelcolor="tab:red")
                ax1.set_xlabel("Number of cross validation folds")
                if prop == "band_gap":
                    ax1.set_ylabel("GP MAE on validation set [eV]", color="red") 
                elif prop == "formation_per_atom" or "e_above_hull":
                    ax1.set_ylabel("GP MAE on validation set [eV/atom]", color="red")
                ax2 = ax1.twinx()
                ax2.plot(mae_test_fold, marker="o", color="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:blue")
                if prop == "band_gap":
                    ax2.set_ylabel("MAE on GP prediction [eV]", color="tab:blue")
                else:
                    ax2.set_ylabel("MAE on GP prediction [eV/atom]", color="tab:blue")

                plt.subplot(425)
                plt.scatter(ytest_dft, gp_mean, c="c",
                            label="R = %.2f, MAE = %.2f, MSE = %.2f, SAE = %.2f"
                            %(R, Optmae, Optmse, Optsae))
                if prop == "band_gap":
                    plt.xlabel("DFT %s [eV]" %prop)
                    plt.ylabel("GP Mean %s [eV]" %prop)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT %s [eV/atom]" %prop)
                    plt.ylabel("GP Mean %s [eV/atom]" %prop)
                plt.legend()
                    
                plt.subplot(426)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]")
                    plt.ylabel("Error in prediction [eV]") 
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]")
                    plt.ylabel("Error in prediction [eV/atom]")
            except:
                plt.subplot(424)
                plt.scatter(ytest_dft, gp_mean, c="c",
                            label="R = %.2f, MAE = %.2f, MSE = %.2f, SAE = %.2f"
                            %(R, Optmae, Optmse, Optsae))
                if prop == "band_gap":
                    plt.xlabel("DFT %s [eV]" %prop)
                    plt.ylabel("GP Mean %s [eV]" %prop)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT %s [eV/atom]" %prop)
                    plt.ylabel("GP Mean %s [eV/atom]" %prop)
                plt.legend()

                plt.subplot(425)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]")
                    plt.ylabel("Error in prediction [eV]")
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]")
                    plt.ylabel("Error in prediction [eV/atom]")                
        else:
            plt.suptitle("GP results of %s layer" %(layer))
            try:
                # Add subplot for k-fold cross validation
                ax1 = plt.subplot(131)
                ax1.plot(best_mae_fold, marker="o", color="tab:red")
                ax1.tick_params(axis="y", labelcolor="tab:red")
                ax1.set_xlabel("Number of cross validation folds")
                if prop == "band_gap":
                    ax1.set_ylabel("GP MAE on validation set [eV]", color="red")
                elif prop == "formation_per_atom" or "e_above_hull":
                    ax1.set_ylabel("GP MAE on validation set [eV/atom]", color="red")
                ax2 = ax1.twinx()
                ax2.plot(mae_test_fold, marker="o", color="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:blue")
                if prop == "band_gap":
                    ax2.set_ylabel("MAE on GP prediction [eV]", color="tab:blue")
                else:
                    ax2.set_ylabel("MAE on GP prediction [eV/atom]", color="tab:blue")
                    
                plt.subplot(132)
                plt.scatter(ytest_dft, gp_mean, c="c",
                            label="R = %.2f, MAE = %.2f, MSE = %.2f, SAE = %.2f"
                            %(R, Optmae, Optmse, Optsae))
                if prop == "band_gap":
                    plt.xlabel("DFT %s [eV]" %prop)
                    plt.ylabel("GP Mean %s [eV]" %prop)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT %s [eV/atom]" %prop)
                    plt.ylabel("GP Mean %s [eV/atom]" %prop)
                plt.legend()
                    
                plt.subplot(133)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]")
                    plt.ylabel("Error in prediction [eV]")
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]")
                    plt.ylabel("Error in prediction [eV/atom]")
            except:
                plt.subplot(121)
                plt.scatter(ytest_dft, gp_mean, c="c",
                            label="R = %.2f, MAE = %.2f, MSE = %.2f, SAE = %.2f"
                            %(R, Optmae, Optmse, Optsae))
                if prop == "band_gap":
                    plt.xlabel("DFT %s [eV]" %prop)
                    plt.ylabel("GP Mean %s [eV]" %prop)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT %s [eV/atom]" %prop)
                    plt.ylabel("GP Mean %s [eV/atom]" %prop)
                plt.legend()

                plt.subplot(122)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]")
                    plt.ylabel("Error in prediction [eV]")
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]")
                    plt.ylabel("Error in prediction [eV/atom]")
                
        plt.savefig("%s/HyperParam_%s.pdf" %(datadir, prop))


    def active(datadir, prop, layer, maxiters, rate, OptLoss, OptAmp, OptLength,
               sampling, query, training_data, ytest_dft, gp_mean, gp_stddev,
               Optmae_val_cycle, mae_test_cycle, mae_test, mse_test, sae_test, R):
        """
        plot.active(datadir, prop, layer, maxiters, rate, OptLoss, OptAmp, OptLength, 
                    sampling, query, training_data, ytest_dft, gp_mean, gp_stddev, 
                    Optmae_val_cycle, mae_test_cycle, mae_test, mse_test, sae_test, R)

        Produces figures after performing active learning. 

        Inputs:
        datadir-              Directory into which results are written into. 
        prop-                 Optical property of interest.  
        layer-                Neural network layer of interest. 
        maxiters-             Number of iterations for optimising hyperparameters.  
        rate-                 Learning rate for Adam optimisation. 
        OptLoss-              Loss minimised during GP training. 
        OptAmp-               Optimised kernel amplitude. 
        OptLength-            Optimised kernel scale length.
        sampling-             Type of sampling for moving data from the test set 
                              into the training set. 
        query-                Number of samples to move from the test set into the 
                              training set.
        training_data-        Length of training data.
        ytest_dft-            DFT-calculated set for testing. 
        gp_mean-              GP predicted optical property values. 
        gp_stddev-            Standard deviation on GP prediction. 
        Optmae_val_cycle-     MAE on validation set. 
        mae_test_cycle-       MAE on GP prediction.
        mae_test-             Mean absolute error.
        mse_test-             Mean squared error.    
        sae_test-             Standard deviation on the absolute error.
        R-                    Pearson correlation coefficient.    

        Outputs:
        1-                    Figure of the MAE on validation sets and 
                              GP prediction vs number of active learning
                              iterations. 
        """
        plt.figure(figsize=[30, 20])
        plt.suptitle("Type of sampling: %s \nAdam optimisation at learning rate %s \nSamples per query = %s"
                     %(sampling, rate, query))
        if maxiters > 0:
            plt.suptitle("GP results of %s layer: Adam optimisation at learning rate %s"
                         %(layer, rate))
            plt.subplot(421)
            plt.plot(OptLoss, "r", label="DFT")
            plt.xlabel("Iterations")
            plt.ylabel("%s Loss" %prop)
            plt.legend()

            plt.subplot(422)
            plt.plot(OptAmp, OptLoss, "b", label="DFT")
            plt.xlabel("Amplitude of the GP kernel")
            plt.ylabel("%s Loss" %prop)
            plt.legend()

            plt.subplot(423)
            plt.plot(OptLength, OptLoss, "g", label="DFT")
            plt.xlabel("Width of the GP kernel")
            plt.ylabel("%s Loss" %prop)
            plt.legend()

            ax1 = plt.subplot(424)
            ax1.plot(training_data, Optmae_val_cycle, marker="o", color="tab:red")
            ax1.tick_params(axis="y", labelcolor="tab:red")
            ax1.set_xlabel("Amount of training data")
            if prop == "band_gap":
                ax1.set_ylabel("GP MAE on validation set [eV]", color="tab:red")
            elif prop == "formation_per_atom" or "e_above_hull":
                ax1.set_ylabel("GP MAE on validation set [eV/atom]", color="tab:red")
            ax2 = ax1.twinx()
            ax2.plot(training_data, mae_test_cycle, marker="o", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            if prop == "band_gap":
                ax2.set_ylabel("MAE on GP prediction [eV]", color="tab:blue")
            else:
                ax2.set_ylabel("MAE on GP prediction [eV/atom]", color="tab:blue") 
                
            plt.subplot(425)
            plt.scatter(ytest_dft, gp_mean, c="c",
                        label="R = %.2f, MAE = %.2f, MSE = %.2f, SAE = %.2f"
                        %(R, mae_test, mse_test, sae_test))
            if prop == "band_gap":
                plt.xlabel("DFT %s [eV]" %prop)
                plt.ylabel("GP Mean %s [eV]" %prop)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("DFT %s [eV/atom]" %prop)
                plt.ylabel("GP Mean %s [eV/atom]" %prop)
            plt.legend()

            plt.subplot(426)
            plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
            if prop == "band_gap":
                plt.xlabel("Uncertainty in prediction [eV]")
                plt.ylabel("Error in prediction [eV]")
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("Uncertainty in prediction [eV/atom]")
                plt.ylabel("Error in prediction [eV/atom]")
        else:
            plt.suptitle("GP results of %s layer" %(layer))
            plt.subplot(121)
            plt.scatter(ytest_dft, gp_mean, c="c",
                        label="R = %.2f, MAE = %.2f, MSE = %.2f, SAE = %.2f"
                        %(R, mae_test, mse_test, sae_test))
            if prop == "band_gap":
                plt.xlabel("DFT %s [eV]" %prop)
                plt.ylabel("GP Mean %s [eV]" %prop)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("DFT %s [eV/atom]" %prop)
                plt.ylabel("GP Mean %s [eV/atom]" %prop)
            plt.legend()

            plt.subplot(122)
            plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
            if prop == "band_gap":
                plt.xlabel("Uncertainty in prediction [eV]")
                plt.ylabel("Error in prediction [eV]")
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("Uncertainty in prediction [eV/atom]")
                plt.ylabel("Error in prediction [eV/atom]")
            
        plt.savefig("%s/active_learn_%s.pdf" %(datadir, prop)) 
