"""
gp-net.py, SciML-SCD, RAL 

For producing plots to inspect the performance of the 
active learning experiment. The plots include those 
produced by Tran et. al, 2020 i.e parity, sharpness, 
dispersion, and calibration plots. The lines of code for 
generating the Tran plots were extracted from Tran et al. 
github repository. 
"""
import random
import numpy as np
import seaborn as sns
from sklearn.metrics import median_absolute_error
from scipy import stats
norm = stats.norm(loc=0, scale=1)
np.random.seed(1)

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
                          hyperparameters obtained from the k-fold cross validation. 
        Optmae-           Mean absolute error.
        Optmse-           Mean squared error.
        Optsae-           Standard deviation on the absolute error. 
        R-                Pearson correlation coefficient.

        Output:
        1-                Figure showing the error in prediction 
                          vs the uncertainty in the prediction. 
        """
        residuals = ytest_dft - gp_mean 
        rmse = np.sqrt(Optmse)
        mdae = median_absolute_error(ytest_dft, gp_mean)
        marpd = np.abs(2 * residuals /
                       (np.abs(gp_mean) + np.abs(ytest_dft))
                       ).mean() * 100
        sharpness = np.sqrt(np.mean(gp_stddev**2))
        dispersion = np.sqrt(((gp_stddev - gp_stddev.mean())**2).mean()) / gp_stddev.mean()

        # For parity and sharpness plots
        if min(ytest_dft) < min(gp_mean):
            lower_lim = min(gp_mean)
            if lower_lim > 0:
                lower_lim = - lower_lim
        else:
            lower_lim = min(ytest_dft)
            if lower_lim > 0:
                lower_lim = - lower_lim
        if max(ytest_dft) < max(gp_mean):
            upper_lim = max(gp_mean)
        else:
            upper_lim = max(ytest_dft)
        lims = [lower_lim-0.5, upper_lim+0.5]
        
        if prop == "band_gap":
            text = ("  MDAE = %.2f eV\n" %mdae +
                    "  MAE = %.2f eV\n" %Optmae +
                    "  SAE = %.2f ev\n" %Optsae +
                    "  RMSE = %.2f eV\n" %rmse +
                    "  MARPD = %i%%\n" %marpd +
                    "  R = %.2f" %R )
            text_hist = "\nSharpness = %.2f eV \nDispersion = %.2f" %(sharpness, dispersion)
        elif prop == "formation_per_atom" or "e_above_hull":
            text = ("  MDAE = %.2f eV/atom\n" %mdae +
                    "  MAE = %.2f eV/atom\n" %Optmae +
                    "  SAE = %.2f ev/atom\n" %Optsae +
                    "  RMSE = %.2f eV/atom\n" %rmse +
                    "  MARPD = %i%%\n" %marpd +
                    "  R = %.2f" %R )
            text_hist = "\nSharpness = %.2f eV/atom \nDispersion = %.2f" %(sharpness, dispersion)

        # For calibration curve
        fontsize = 18
        rc = {"font.size": fontsize,
              "axes.labelsize": fontsize, 
              "axes.titlesize": fontsize, 
              "xtick.labelsize": fontsize, 
              "ytick.labelsize": fontsize, 
              "legend.fontsize": fontsize}
                 
        # Compute the calibration curve 
        def calculate_density(percentile):
            """
            Calculates the fraction of the residuals that fall within the 
            lower percentile of their respectie Gaussian distributions,
            which are defined by their respective uncertainty estimates
            """
            # Find the normalized bounds of this percentile
            upper_bound = norm.ppf(percentile)

            # Normalize the residuals so they all should fall on the normal bell curve
            normalized_residuals = residuals / gp_stddev

            # Count how many residuals fall inside here
            num_within_quantile = 0
            for resid in normalized_residuals:
                if resid <= upper_bound:
                    num_within_quantile += 1

            # Return the fraction of residuals that fall within the bounds
            density = num_within_quantile / len(residuals)
            return density
        
        predicted_pi = np.linspace(0, 1, 100)
        observed_pi = np.array([calculate_density(quantile) for quantile in predicted_pi])
        calibration_error = ((predicted_pi - observed_pi)**2).sum()
        miscalibration_area = np.mean(np.abs(predicted_pi - observed_pi))     

        # Errorbar plot
        num_samples = 200
        all_predictions = list(zip(gp_mean, ytest_dft, gp_stddev))
        samples = random.sample(all_predictions, k=num_samples)
        _preds, _targets, _stddevs = zip(*samples)
        _preds = np.array(_preds)
        _targets = np.array(_targets)
        _stddevs = np.array(_stddevs)
        

        plt.figure(figsize=[30, 20])
        plt.suptitle("GP results of %s layer for %s" %(layer, prop), fontsize=20)
        if maxiters > 0:
            plt.subplot(421)
            plt.plot(OptLoss, "r") 
            plt.xlabel("Iterations", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize) 

            plt.subplot(422)
            plt.plot(OptAmp, OptLoss, "b") 
            plt.xlabel("Amplitude of the GP kernel", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize) 
            
            plt.subplot(423)
            plt.plot(OptLength, OptLoss, "g") 
            plt.xlabel("Width of the GP kernel", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize) 
            try:
                # Add subplot for k-fold cross validation 
                ax1 = plt.subplot(424)
                ax1.plot(best_mae_fold, marker="o", color="tab:red")
                ax1.tick_params(axis="y", labelcolor="tab:red")
                ax1.set_xlabel("Number of cross validation folds", fontsize=fontsize)
                if prop == "band_gap":
                    ax1.set_ylabel("MAE on validation set [eV]", color="red",
                                   fontsize=fontsize) 
                elif prop == "formation_per_atom" or "e_above_hull":
                    ax1.set_ylabel("MAE on validation set [eV/atom]", color="red",
                                   fontsize=fontsize)
                ax2 = ax1.twinx()
                ax2.plot(mae_test_fold, marker="o", color="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:blue")
                if prop == "band_gap":
                    ax2.set_ylabel("MAE on prediction [eV]", color="tab:blue",
                                   fontsize=fontsize)
                else:
                    ax2.set_ylabel("MAE on prediction [eV/atom]", color="tab:blue",
                                   fontsize=fontsize)
                    
                plt.subplot(425)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV]", fontsize=fontsize) 
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)
                    
                plt.subplot(426)
                plt.plot(lims, lims, "--")
                plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
                plt.xlim(lims)
                plt.ylim(lims)
                plt.text(x=lims[0], y=lims[1], s=text,
                         horizontalalignment="left",
                         verticalalignment="top",
                         fontsize=16) 
                if prop == "band_gap":
                    plt.xlabel("DFT [eV]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
                plt.legend(loc=1)

                plt.subplot(427)
                ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
                ax_sharp.set_xlim([0, 1])
                ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
                ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

                ax_sharp.axvline(x=sharpness, label="sharpness")
                ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                              s=text_hist,
                              verticalalignment="top",
                              fontsize=fontsize)
                
                plt.subplot(428)
                sns.set(rc=rc)
                sns.set_style("ticks")
                ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
                ax_ideal.lines[0].set_linestyle("--")
                ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
                ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                           alpha=0.2, label="miscalibration area")
                ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
                ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
                ax_ideal.set_xlim([0, 1])
                ax_ideal.set_ylim([0, 1])
                plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                         % (calibration_error, miscalibration_area),
                         verticalalignment="bottom",
                         horizontalalignment="right",
                         fontsize=fontsize)
            except:
                plt.subplot(424)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)
                    
                plt.subplot(425)
                plt.plot(lims, lims, "--")                
                plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
                plt.xlim(lims)
                plt.ylim(lims)
                plt.text(x=lims[0], y=lims[1], s=text,
                         horizontalalignment="left",
                         verticalalignment="top",
                         fontsize=16)                
                if prop == "band_gap":
                    plt.xlabel("DFT [eV]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
                plt.legend(loc=1)
                    
                plt.subplot(426)
                ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
                ax_sharp.set_xlim([0, 1])
                ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
                ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

                ax_sharp.axvline(x=sharpness, label="sharpness")
                ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                              s=text_hist,
                              verticalalignment="top",
                              fontsize=fontsize)
                
                plt.subplot(427)
                sns.set(rc=rc)
                sns.set_style("ticks")
                ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
                ax_ideal.lines[0].set_linestyle("--")
                ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
                ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                           alpha=0.2, label="miscalibration area")
                ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
                ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize) 
                ax_ideal.set_xlim([0, 1])
                ax_ideal.set_ylim([0, 1])
                plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                         % (calibration_error, miscalibration_area),
                         verticalalignment="bottom",
                         horizontalalignment="right",
                         fontsize=fontsize)
        else:
            try:
                # Add subplot for k-fold cross validation
                ax1 = plt.subplot(321)
                ax1.plot(best_mae_fold, marker="o", color="tab:red")
                ax1.tick_params(axis="y", labelcolor="tab:red")
                ax1.set_xlabel("Number of cross validation folds")
                if prop == "band_gap":
                    ax1.set_ylabel("MAE on validation set [eV]", color="red",
                                   fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    ax1.set_ylabel("MAE on validation set [eV/atom]", color="red",
                                   fontsize=fontsize)
                ax2 = ax1.twinx()
                ax2.plot(mae_test_fold, marker="o", color="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:blue")
                if prop == "band_gap":
                    ax2.set_ylabel("MAE on GP prediction [eV]", color="tab:blue",
                                   fontsize=fontsize)
                else:
                    ax2.set_ylabel("MAE on GP prediction [eV/atom]", color="tab:blue",
                                   fontsize=fontsize)

                plt.subplot(322)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)
                    
                plt.subplot(323)
                plt.plot(lims, lims, "--")
                plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
                plt.xlim(lims)
                plt.ylim(lims)
                plt.text(x=lims[0], y=lims[1], s=text,
                         horizontalalignment="left",
                         verticalalignment="top",
                          fontsize=16)
                if prop == "band_gap":
                    plt.xlabel("DFT [eV]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize) 
                plt.legend(loc=1)
                    
                plt.subplot(324)
                ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
                ax_sharp.set_xlim([0, 1])
                ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
                ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

                ax_sharp.axvline(x=sharpness, label="sharpness")
                ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                              s=text_hist,
                              verticalalignment="top",
                              fontsize=fontsize)
                
                plt.subplot(325)
                sns.set(rc=rc)
                sns.set_style("ticks")
                ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
                ax_ideal.lines[0].set_linestyle("--")
                ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
                ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                           alpha=0.2, label="miscalibration area")
                ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
                ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
                ax_ideal.set_xlim([0, 1])
                ax_ideal.set_ylim([0, 1])
                plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                         % (calibration_error, miscalibration_area),
                         verticalalignment="bottom",
                         horizontalalignment="right",
                         fontsize=fontsize)
            except:
                plt.subplot(221)
                plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
                if prop == "band_gap":
                    plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                    plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)

                plt.subplot(222)
                plt.plot(lims, lims, "--")
                plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
                plt.xlim(lims)
                plt.ylim(lims)                
                if prop == "band_gap":
                    plt.xlabel("DFT [eV]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV]", fontsize=fontsize)
                elif prop == "formation_per_atom" or "e_above_hull":
                    plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                    plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
                plt.text(x=lims[0], y=lims[1], s=text,
                         horizontalalignment="left",
                         verticalalignment="top",
                         fontsize=16)                    
                plt.legend(loc=1)                    
                    
                plt.subplot(223)
                ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
                ax_sharp.set_xlim([0, 1])
                ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
                ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

                ax_sharp.axvline(x=sharpness, label="sharpness")
                ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                              s=text_hist,
                              verticalalignment="top",
                              fontsize=fontsize)
                
                plt.subplot(224)
                sns.set(rc=rc)
                sns.set_style("ticks")
                ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
                ax_ideal.lines[0].set_linestyle("--")
                ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
                ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                           alpha=0.2, label="miscalibration area")
                ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
                ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
                ax_ideal.set_xlim([0, 1])
                ax_ideal.set_ylim([0, 1])
                plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                         % (calibration_error, miscalibration_area),
                         verticalalignment="bottom",
                         horizontalalignment="right",
                         fontsize=fontsize)

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
        1-                    Figures showing the performance of the active learning
                              experiment.
        """
        residuals = ytest_dft - gp_mean
        rmse = np.sqrt(mse_test)
        mdae = median_absolute_error(ytest_dft, gp_mean)
        marpd = np.abs(2 * residuals /
                       (np.abs(gp_mean) + np.abs(ytest_dft))
                        ).mean() * 100
        sharpness = np.sqrt(np.mean(gp_stddev**2))
        dispersion = np.sqrt(((gp_stddev - gp_stddev.mean())**2).mean()) / gp_stddev.mean()

        # For parity and sharpness plots
        if min(ytest_dft) < min(gp_mean):
            lower_lim = min(gp_mean)
            if lower_lim > 0:
                lower_lim = - lower_lim
        else:
            lower_lim = min(ytest_dft)
            if lower_lim > 0:
                lower_lim = - lower_lim
        if max(ytest_dft) < max(gp_mean):
            upper_lim = max(gp_mean)
        else:
            upper_lim = max(ytest_dft)
        lims = [lower_lim-0.5, upper_lim+0.5]
        
        if prop == "band_gap":
            text = ("  MDAE = %.2f eV\n" %mdae +
                    "  MAE = %.2f eV\n" %mae_test +
                    "  SAE = %.2f ev\n" %sae_test +
                    "  RMSE = %.2f eV\n" %rmse +
                    "  MARPD = %i%%\n" %marpd +
                    "  R = %.2f" %R )
            text_hist = "\nSharpness = %.2f eV \nDispersion = %.2f" % (sharpness, dispersion)
        elif prop == "formation_per_atom" or "e_above_hull":
            text = ("  MDAE = %.2f eV/atom\n" %mdae +
                    "  MAE = %.2f eV/atom\n" %mae_test +
                    "  SAE = %.2f ev/atom\n" %sae_test +
                    "  RMSE = %.2f eV/atom\n" %rmse +
                    "  MARPD = %i%%\n" %marpd +
                    "  R = %.2f" %R )
            text_hist = "\nSharpness = %.2f eV/atom \nDispersion = %.2f" % (sharpness, dispersion)        
            
        # For calibration curve
        fontsize = 18
        rc = {"font.size": fontsize,
              "axes.labelsize": fontsize,
              "axes.titlesize": fontsize,
              "xtick.labelsize": fontsize,
              "ytick.labelsize": fontsize,
              "legend.fontsize": fontsize}

        # Compute the calibration curve
        def calculate_density(percentile):
            # Find the normalized bounds of this percentile
            upper_bound = norm.ppf(percentile)

            # Normalize the residuals so they all should fall on the normal bell curve
            normalized_residuals = residuals / gp_stddev

            # Count how many residuals fall inside here
            num_within_quantile = 0
            for resid in normalized_residuals:
                if resid <= upper_bound:
                    num_within_quantile += 1

            # Return the fraction of residuals that fall within the bounds
            density = num_within_quantile / len(residuals)
            return density
        
        predicted_pi = np.linspace(0, 1, 100) 
        observed_pi = np.array([calculate_density(quantile) for quantile in predicted_pi])
        calibration_error = ((predicted_pi - observed_pi)**2).sum()
        miscalibration_area = np.mean(np.abs(predicted_pi - observed_pi))

        # Errorbar plot
        num_samples = 200
        all_predictions = list(zip(gp_mean, ytest_dft, gp_stddev))
        samples = random.sample(all_predictions, k=num_samples)
        _preds, _targets, _stddevs = zip(*samples)
        _preds = np.array(_preds)
        _targets = np.array(_targets)
        _stddevs = np.array(_stddevs)  

        
        plt.figure(figsize=[30, 20])
        plt.suptitle("GP results of %s layer for %s \nType of sampling: %s \nSamples per query = %s"
                     %(layer, prop, sampling, query), fontsize=20)
        if maxiters > 0:
            plt.subplot(421)
            plt.plot(OptLoss, "r")
            plt.xlabel("Iterations", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize) 

            plt.subplot(422)
            plt.plot(OptAmp, OptLoss, "b")
            plt.xlabel("Amplitude of the GP kernel", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize) 

            plt.subplot(423)
            plt.plot(OptLength, OptLoss, "g")
            plt.xlabel("Width of the GP kernel", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize) 

            ax1 = plt.subplot(424)
            ax1.plot(training_data, Optmae_val_cycle, marker="o", color="tab:red")
            ax1.tick_params(axis="y", labelcolor="tab:red")
            ax1.set_xlabel("Amount of training data", fontsize=fontsize)
            if prop == "band_gap":
                ax1.set_ylabel("MAE on validation set [eV]", color="tab:red",
                               fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                ax1.set_ylabel("MAE on validation set [eV/atom]", color="tab:red",
                               fontsize=fontsize)
            ax2 = ax1.twinx()
            ax2.plot(training_data, mae_test_cycle, marker="o", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            if prop == "band_gap":
                ax2.set_ylabel("MAE on GP prediction [eV]", color="tab:blue",
                               fontsize=fontsize)
            else:
                ax2.set_ylabel("MAE on GP prediction [eV/atom]", color="tab:blue",
                               fontsize=fontsize) 

            plt.subplot(425)
            plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
            if prop == "band_gap":
                plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                plt.ylabel("Residuals [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)

            plt.subplot(426)
            plt.plot(lims, lims, "--")
            plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.text(x=lims[0], y=lims[1], s=text,
                     horizontalalignment="left",
                     verticalalignment="top",
                     fontsize=16)
            if prop == "band_gap":
                plt.xlabel("DFT [eV]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
            plt.legend(loc=1)

            plt.subplot(427)
            ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
            ax_sharp.set_xlim([0, 1])
            ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
            ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

            ax_sharp.axvline(x=sharpness, label="sharpness")
            ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                          s=text_hist,
                          verticalalignment="top",
                          fontsize=fontsize)

            plt.subplot(428)
            sns.set(rc=rc)
            sns.set_style("ticks")
            ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
            ax_ideal.lines[0].set_linestyle("--")
            ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
            ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                       alpha=0.2, label="miscalibration area")
            ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
            ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
            ax_ideal.set_xlim([0, 1])
            ax_ideal.set_ylim([0, 1])
            plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                      % (calibration_error, miscalibration_area),
                     verticalalignment="bottom",
                     horizontalalignment="right",
                     fontsize=fontsize)
        else:
            plt.subplot(221)
            plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
            if prop == "band_gap":
                plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                plt.ylabel("Residuals [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)
            
            plt.subplot(222)
            plt.plot(lims, lims, "--")
            plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.text(x=lims[0], y=lims[1], s=text,
                     horizontalalignment="left",
                     verticalalignment="top",
                     fontsize=16)
            if prop == "band_gap":
                plt.xlabel("DFT [eV]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
            plt.legend(loc=1)

            plt.subplot(223)
            ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
            ax_sharp.set_xlim([0, 1])
            ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
            ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

            ax_sharp.axvline(x=sharpness, label="sharpness")
            ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                          s=text_hist,
                          verticalalignment="top",
                          fontsize=fontsize)

            plt.subplot(224)
            sns.set(rc=rc)
            sns.set_style("ticks")
            ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
            ax_ideal.lines[0].set_linestyle("--")
            ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
            ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                       alpha=0.2, label="miscalibration area")
            ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
            ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
            ax_ideal.set_xlim([0, 1])
            ax_ideal.set_ylim([0, 1])
            plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                     % (calibration_error, miscalibration_area),
                     verticalalignment="bottom",
                     horizontalalignment="right",
                     fontsize=fontsize)
                
        plt.savefig("%s/active_learn_%s.pdf" %(datadir, prop)) 


    def norepeat(datadir, prop, layer, sampling, query, maxiters):
        """
        plot.norepeat(datadir, prop, layer, sampling, query, maxiters)

        Produces figures after performing norepeat active learning.

        Inputs:
        datadir-            Directory into which results are written into. 
        prop-               Optical property of interest. 
        layer-              Neural network layer of interest.  
        sampling-           Type of sampling for moving data from the test set
                            into the training set. 
        query-              Number of samples to move from the test set into the 
                            into the training set. 
        maxiters-           Number of iterations for optimising hyperparameters.

        Outputs:
        1-                  Figures showing the performance of the active learning 
                            experiment.
        """
        import os 
        from scipy.stats import pearsonr

        ytest_dft = np.load("%s/ytest.npy" %datadir)
        gp_mean = np.load("%s/gp_mean.npy" %datadir)
        gp_stddev = np.load("%s/gp_stddev.npy" %datadir)
        mae_test_cycle = np.load("%s/gp_mae.npy" %datadir)
        mse_test_cycle = np.load("%s/gp_mse.npy" %datadir)[np.argmin(mae_test_cycle)]
        sae_test_cycle  = np.load("%s/gp_sae.npy" %datadir)[np.argmin(mae_test_cycle)]
        training_data = np.load("%s/training_data_for_plotting.npy" %datadir)
        
        residuals = ytest_dft - gp_mean
        rmse = np.sqrt(mse_test_cycle) 
        mdae = median_absolute_error(ytest_dft, gp_mean)
        marpd = np.abs(2 * residuals /
                       (np.abs(gp_mean) + np.abs(ytest_dft))
                        ).mean() * 100
        sharpness = np.sqrt(np.mean(gp_stddev**2))
        dispersion = np.sqrt(((gp_stddev - gp_stddev.mean())**2).mean()) / gp_stddev.mean()
        R, p = pearsonr(x=ytest_dft, y=gp_mean)

        # For parity and sharpness plots
        if min(ytest_dft) < min(gp_mean):
            lower_lim = min(gp_mean)
            if lower_lim > 0:
                lower_lim = - lower_lim
        else:
            lower_lim = min(ytest_dft)
            if lower_lim > 0:
                lower_lim = - lower_lim
        if max(ytest_dft) < max(gp_mean):
            upper_lim = max(gp_mean)
        else:
            upper_lim = max(ytest_dft)
        lims = [lower_lim-0.5, upper_lim+0.5]

        if prop == "band_gap":
            text = ("  MDAE = %.2f eV\n" %mdae +
                    "  MAE = %.2f eV\n" %min(mae_test_cycle) +
                    "  SAE = %.2f ev\n" %sae_test_cycle +
                    "  RMSE = %.2f eV\n" %rmse +
                    "  MARPD = %i%%\n" %marpd +
                    "  R = %.2f" %R )
            text_hist = "\nSharpness = %.2f eV \nDispersion = %.2f" %(sharpness, dispersion)
        elif prop == "formation_per_atom" or "e_above_hull":
            text = ("  MDAE = %.2f eV/atom\n" %mdae +
                    "  MAE = %.2f eV/atom\n" %min(mae_test_cycle) +
                    "  SAE = %.2f ev/atom\n" %sae_test_cycle +
                    "  RMSE = %.2f eV/atom\n" %rmse +
                    "  MARPD = %i%%\n" %marpd +
                    "  R = %.2f" %R )
            text_hist = "\nSharpness = %.2f eV/atom \nDispersion = %.2f" % (sharpness, dispersion)

        # For calibration curve
        fontsize = 18
        rc = {"font.size": fontsize,
              "axes.labelsize": fontsize,
              "axes.titlesize": fontsize,
              "xtick.labelsize": fontsize,
              "ytick.labelsize": fontsize,
              "legend.fontsize": fontsize}

        # Compute the calibration curve
        def calculate_density(percentile):
            """
            Calculates the fraction of the residuals that fall within the 
            lower percentile of their respectie Gaussian distributions,
            which are defined by their respective uncertainty estimates
            """
            # Find the normalized bounds of this percentile
            upper_bound = norm.ppf(percentile)
            
            # Normalize the residuals so they all should fall on the normal bell curve
            normalized_residuals = residuals / gp_stddev
            
            # Count how many residuals fall inside here
            num_within_quantile = 0
            for resid in normalized_residuals:
                if resid <= upper_bound:
                    num_within_quantile += 1

            # Return the fraction of residuals that fall within the bounds
            density = num_within_quantile / len(residuals)
            return density

        predicted_pi = np.linspace(0, 1, 100)
        observed_pi = np.array([calculate_density(quantile) for quantile in predicted_pi])
        calibration_error = ((predicted_pi - observed_pi)**2).sum()
        miscalibration_area = np.mean(np.abs(predicted_pi - observed_pi))        

        # Errorbar plot
        num_samples = 200
        all_predictions = list(zip(gp_mean, ytest_dft, gp_stddev))
        samples = random.sample(all_predictions, k=num_samples)
        _preds, _targets, _stddevs = zip(*samples)
        _preds = np.array(_preds)
        _targets = np.array(_targets)
        _stddevs = np.array(_stddevs)
         
        plt.figure(figsize=[30, 20])
        plt.suptitle("GP results of %s layer for %s \nType of sampling: %s \nSamples per query = %s"
                     %(layer, prop, sampling, query), fontsize=20)
        if os.path.isfile("%s/OptLoss.npy" %datadir):
            OptLoss = np.load("%s/OptLoss.npy" %datadir)
            OptAmp = np.load("%s/OptAmp.npy" %datadir)
            OptLength = np.load("%s/OptLength.npy" %datadir)
            
            plt.subplot(421)
            plt.plot(OptLoss, "r")
            plt.xlabel("Iterations", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize)
            
            plt.subplot(422)
            plt.plot(OptAmp, OptLoss, "b") 
            plt.xlabel("Amplitude of the GP kernel", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize)
            
            plt.subplot(423)
            plt.plot(OptLength, OptLoss, "g") 
            plt.xlabel("Width of the GP kernel", fontsize=fontsize)
            plt.ylabel("Loss", fontsize=fontsize)

            plt.subplot(424) 
            plt.plot(training_data, mae_test_cycle, marker="o", color="tab:blue")
            plt.xlabel("Amount of training data", fontsize=fontsize)
            if prop == "band_gap":
                plt.ylabel("MAE on GP prediction [eV]", color="tab:blue", fontsize=fontsize)
            else:
                plt.ylabel("MAE on GP prediction [eV/atom]", color="tab:blue", fontsize=fontsize)

            plt.subplot(425)
            plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
            if prop == "band_gap":
                plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                plt.ylabel("Residuals [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)
                
            plt.subplot(426)
            plt.plot(lims, lims, "--")
            plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.text(x=lims[0], y=lims[1], s=text,
                     horizontalalignment="left",
                     verticalalignment="top",
                     fontsize=16)
            if prop == "band_gap":
                plt.xlabel("DFT [eV]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
            plt.legend(loc=1)

            plt.subplot(427)
            ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
            ax_sharp.set_xlim([0, 1])
            ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
            ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)

            ax_sharp.axvline(x=sharpness, label="sharpness")
            ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                          s=text_hist,
                          verticalalignment="top",
                          fontsize=fontsize)

            plt.subplot(428)
            sns.set(rc=rc)
            sns.set_style("ticks")
            ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
            ax_ideal.lines[0].set_linestyle("--")
            ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
            ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                       alpha=0.2, label="miscalibration area")
            ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
            ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
            ax_ideal.set_xlim([0, 1])
            ax_ideal.set_ylim([0, 1])
            plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                     %(calibration_error, miscalibration_area),
                     verticalalignment="bottom",
                     horizontalalignment="right",
                     fontsize=fontsize)            
            
        else:
            plt.subplot(321)
            plt.plot(training_data, mae_test_cycle, marker="o", color="r")
            plt.xlabel("Amount of training data", fontsize=fontsize)
            if prop == "band_gap":
                plt.ylabel("MAE on GP prediction [eV]", fontsize=fontsize)
            else:
                plt.ylabel("MAE on GP prediction [eV/atom]", fontsize=fontsize)

            plt.subplot(322)
            plt.scatter(gp_stddev, ytest_dft - gp_mean, c="m")
            if prop == "band_gap":
                plt.xlabel("Uncertainty in prediction [eV]", fontsize=fontsize)
                plt.ylabel("Residuals [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("Uncertainty in prediction [eV/atom]", fontsize=fontsize)
                plt.ylabel("Residuals [eV/atom]", fontsize=fontsize)
                
            plt.subplot(323)
            plt.plot(lims, lims, "--")
            plt.errorbar(_targets, _preds, yerr=3*_stddevs, fmt="o", label="3 $\sigma$")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.text(x=lims[0], y=lims[1], s=text,
                     horizontalalignment="left",
                     verticalalignment="top",
                     fontsize=16)
            if prop == "band_gap":
                plt.xlabel("DFT [eV]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV]", fontsize=fontsize)
            elif prop == "formation_per_atom" or "e_above_hull":
                plt.xlabel("DFT [eV/atom]", fontsize=fontsize)
                plt.ylabel("GP Mean [eV/atom]", fontsize=fontsize)
            plt.legend(loc=1)                

            plt.subplot(324)
            ax_sharp = sns.distplot(gp_stddev, kde=False, norm_hist=True)
            ax_sharp.set_xlim([0, 1])
            ax_sharp.set_xlabel("Predicted standard deviations (eV)", fontsize=fontsize)
            ax_sharp.set_ylabel("Normalized frequency", fontsize=fontsize)
            
            ax_sharp.axvline(x=sharpness, label="sharpness")
            ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                          s=text_hist,
                          verticalalignment="top",
                          fontsize=fontsize)

            plt.subplot(325) 
            sns.set(rc=rc)
            sns.set_style("ticks")
            ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
            ax_ideal.lines[0].set_linestyle("--")
            ax_gp = sns.lineplot(predicted_pi, observed_pi, label="GP")
            ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                                       alpha=0.2, label="miscalibration area")
            ax_ideal.set_xlabel("Expected cumulative distribution", fontsize=fontsize)
            ax_ideal.set_ylabel("Observed cumulative distribution", fontsize=fontsize)
            ax_ideal.set_xlim([0, 1])
            ax_ideal.set_ylim([0, 1])
            plt.text(x=0.95, y=0.05, s="Calibration error = %.3f \nMiscalibration area = %.3f"
                     %(calibration_error, miscalibration_area),
                     verticalalignment="bottom",
                     horizontalalignment="right",
                     fontsize=fontsize)               

        plt.savefig("%s/active_learn_%s.pdf" %(datadir, prop))
