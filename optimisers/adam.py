"""
adam.py, SciML-SCD, RAL

Uses the adam optimiser to optimise the hyperparameters of the Matern
One Half kernel Gaussian Process. This process is also known as the 
Ornstein-Uhlenbeck process. The optical properties of the materials 
are predicted by the GP, and their uncertainties estimated. 
"""
import logging
import os 
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(levelname)s:gp-net: %(message)s")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions 
tfk = tfp.math.psd_kernels
tfb = tfp.bijectors 


def convert_index_points(array):
    """
    Reshape an array into a tensor appropriate for GP index points.
    Extends the amount of dimensions by `array.shape[1] - 1` and 
    converts to a `Tensor` with `dtype=tf.float64`.

    Inputs:
    array-        The array to extend.

    Outputs:
    1-            The converted Tensor.
    """
    shape = array.shape
    shape += (1,) * (shape[1] - 1)
    return tf.constant(array, dtype=tf.float64, shape=shape), shape[1]


class adam:
        
    def train_test_split(datadir, prop, tsne_pool, tsne_test, ypool_dft,
                         ytest_dft, maxiters, amp, length_scale, rate):
        """
        adam.train_test_split(datadir, prop, tsne_pool, tsne_test, ypool_dft, 
                              ytest_dft, maxiters, amp, length_scale, rate)

        A Gaussian Process (GP) with a Matern One Half kernel is used in the 
        case where a train-test data split approach is used. 

        Inputs:
        datadir-            Directory into which results are written into.
        prop-               Optical property of interest.
        tsne_pool-          Latent points for the pool.
        tsne_test-          Latent points for the test set.
        ypool_dft-          DFT-calculated set in the pool.
        ytest_dft-          DFT-calculated set for testing.
        maxiters-           Number of iterations for optimising hyperparameters.
        amp-                Maximum value of the kernel.
        length_scale-       The width of the kernel.  
        rate                Learning rate for Adam optimisation. 
        
        Outputs:
        1-                  Optimised loss.
        2-                  Optimised kernel amplitude.
        3-                  Optimised kernel length scale. 
        4-                  Best MAE and its corresponding MSE, the 
                            standard deviation on the MAE on the test set.
        5-                  GP prediction.
        6-                  Uncertainty on the GP prediction.
        7-                  Pearson correlation coefficient between the DFT-
                            calculated and GP-predicted optical property. 
        """
        latent_pool = convert_index_points(tsne_pool)[0]
        latent_test = convert_index_points(tsne_test)[0]
        feature_ndims = convert_index_points(tsne_pool)[1]

        # Define the DFT-calculated values
        ypool_dft = tf.constant(ypool_dft, dtype=tf.float64)
        ytest_dft = tf.constant(ytest_dft, dtype=tf.float64)
            
        if maxiters <= 0: 
            amp = tf.cast(amp, tf.float64)
            length_scale = tf.cast(length_scale, tf.float64)
            print("Prior on the amplitude of the kernel = %.4f" %amp.numpy())
            print("Prior on the width of the kernel = %.4f" %length_scale.numpy())
            logging.info("No bijector is applied to the priors ...")

            gprm_dft = tfd.GaussianProcessRegressionModel(
                kernel=tfk.MaternOneHalf(amp,
                                         length_scale,
                                         feature_ndims=feature_ndims), 
                index_points=latent_test,
                observation_index_points=latent_pool,
                observations=ypool_dft)
        else:
            print("Requested optimisation with Adam algorithm at learning rate %s" %rate)
            print("Number of iterations = %s" %maxiters)
            print("Prior on the amplitude of the kernel = %s" %amp)
            print("Prior on the width of the kernel = %s" %length_scale)
            optimizer = tf.optimizers.Adam(learning_rate=rate)

            # Create trainable variables and apply positive constraint
            amp = tfp.util.TransformedVariable(initial_value=amp,
                                               bijector=tfb.Exp(), 
                                               name="amp",
                                               dtype=tf.float64)
            length_scale = tfp.util.TransformedVariable(initial_value=length_scale,
                                                        bijector=tfb.Exp(), 
                                                        name="length_scale",
                                                        dtype=tf.float64)

            def trainables():
                return [var.trainable_variables[0] for var in [amp, length_scale]]

            logging.info("Training GP on the pool to minimise MAE on the test set ...")
            @tf.function
            def loss_fn():
                """ The loss function to be minimised for the DFT values in the pool """
                kernel = tfk.MaternOneHalf(amp, 
                                           length_scale,
                                           feature_ndims=feature_ndims) 
                gp = tfd.GaussianProcess(kernel=kernel, index_points=latent_pool)
                return -gp.log_prob(ypool_dft)

            OptLoss = np.array([ ])
            OptAmp = np.array([ ]) 
            OptLength = np.array([ ]) 
            Optmae = np.array([ ]) 
            Optmse = np.array([ ]) 
            Optsae = np.array([ ]) 
            for i in tf.range(maxiters):
                with tf.GradientTape() as tape:
                    loss = loss_fn()
                grads = tape.gradient(loss, trainables())
                optimizer.apply_gradients(zip(grads, trainables()))
                OptLoss = np.append(OptLoss, loss.numpy())
                OptAmp = np.append(OptAmp, amp._value().numpy())
                OptLength = np.append(OptLength, length_scale._value().numpy())
                gprm_dft = tfd.GaussianProcessRegressionModel(
                    kernel=tfk.MaternOneHalf(OptAmp[i],
                                             OptLength[i],
                                             feature_ndims=feature_ndims),
                    index_points=latent_test,
                    observation_index_points=latent_pool,
                    observations=ypool_dft)
                Optmae = np.append(Optmae, tf.losses.MAE(ytest_dft, gprm_dft.mean()).numpy())
                Optmse = np.append(Optmse, tf.losses.MSE(ytest_dft, gprm_dft.mean()).numpy())
                Optsae = np.append(Optsae, np.std(np.abs(gprm_dft.mean().numpy() - ytest_dft.numpy())))
                if i % 10 == 0 or i + 1 == maxiters:
                    print("At step %d: loss=%.4f, amplitude=%.4f, length_scale=%.4f, mae=%.4f, mse=%.4f, sae=%.4f, min(std)=%.4f, max(std)=%.4f"
                          %(i, OptLoss[i], OptAmp[i], OptLength[i], Optmae[i], Optmse[i], Optsae[i],
                            min(gprm_dft.stddev().numpy()), max(gprm_dft.stddev().numpy())))
                    
        # Compute the Pearson correlation coefficient 
        R, p = pearsonr(x=ytest_dft.numpy(), y=gprm_dft.mean().numpy())

        logging.info("Writing results to file ...")
        if maxiters > 0:
            np.save("%s/OptLoss.npy" %datadir, OptLoss) 
            np.save("%s/OptAmp.npy" %datadir, OptAmp) 
            np.save("%s/OptLength.npy" %datadir, OptLength) 
            np.save("%s/Optmae.npy" %datadir, Optmae) 
            np.save("%s/Optmse.npy" %datadir, Optmse)
            np.save("%s/Optsae.npy" %datadir, Optsae)
        np.save("%s/ypool.npy" %datadir, ypool_dft.numpy())
        np.save("%s/ytest.npy" %datadir, ytest_dft.numpy())
        np.save("%s/gp_mean.npy" %datadir, gprm_dft.mean().numpy())
        np.save("%s/gp_stddev.npy" %datadir, gprm_dft.stddev().numpy())
        np.save("%s/gp_variance.npy" %datadir, gprm_dft.variance().numpy())

        if maxiters <= 0:
            print("\nPrediction statistics: mae = %.4f, mse = %.4f, sae = %.4f, min(std) = %.4f, max(std) = %.4f, R = %.4f"
                  %(tf.losses.MAE(ytest_dft, gprm_dft.mean()).numpy(),
                    tf.losses.MSE(ytest_dft, gprm_dft.mean()).numpy(),
                    np.std(np.abs(gprm_dft.mean().numpy() - ytest_dft.numpy())),
                    min(gprm_dft.stddev().numpy()),
                    max(gprm_dft.stddev().numpy()),
                    R) )
            
            return ( None,
                     amp.numpy(),
                     length_scale.numpy(),
                     tf.losses.MAE(ytest_dft, gprm_dft.mean()).numpy(),
                     tf.losses.MSE(ytest_dft, gprm_dft.mean()).numpy(),
                     np.std(np.abs(gprm_dft.mean().numpy() - ytest_dft.numpy())),
                     gprm_dft.mean().numpy(),
                     gprm_dft.stddev().numpy(),
                     R )
        else:
            logging.info("Best-fitted parameters:")
            print("          amplitude: %.4f" %OptAmp[np.argmin(Optmae)])
            print("          length_scale: %.4f" %OptLength[np.argmin(Optmae)])
            print("          Prediction statistics: mae = %.4f, mse = %.4f, sae = %.4f, min(std) = %.4f, max(std) = %.4f, R = %.4f"
                  %(min(Optmae),
                    Optmse[np.argmin(Optmae)],
                    Optsae[np.argmin(Optmae)],
                    min(gprm_dft.stddev().numpy()),
                    max(gprm_dft.stddev().numpy()),
                    R) )
            
            return ( OptLoss,
                     OptAmp,
                     OptLength,
                     min(Optmae), 
                     Optmse[np.argmin(Optmae)], 
                     Optsae[np.argmin(Optmae)], 
                     gprm_dft.mean().numpy(),
                     gprm_dft.stddev().numpy(),
                     R )


    def k_fold(datadir, prop, tsne_train, tsne_val, tsne_test, ytrain_dft,
               yval_dft, ytest_dft, maxiters, amp, length_scale, rate):
        """ 
        adam.k_fold(datadir, prop, tsne_train, tsne_val, tsne_test,
                    ytrain_dft, yval_dft, ytest_dft, maxiters, amp, 
                    length_scale, rate) 

        k-fold cross-validation Gaussian Process with a Matern One Half kernel. 

        Inputs:
        datadir-            Directory into which results are written into. 
        prop-               Optical property of interest.
        tsne_train-         Latent points for the training set. 
        tsne_val-           Latent points for the validation set. 
        tsne_test-          Latent points for the test set. 
        ytrain_dft-         DFT-calculated set for training.  
        yval_dft-           DFT-calculated set for validation. 
        ytest_dft-          DFT-calculated set for testing.  
        maxiters-           Number of iterations for optimising hyperparameters.
        amp-                Maximum value of the kernel.
        length_scale-       The width of the kernel. 
        rate-               Learning rate for Adam optimisation.

        Outputs:
        1-                  Best amplitude. 
        2-                  Best length scale. 
        3-                  Best MAE and it's corresponding MSE on the validation 
                            set.
        4-                  MSE on the test set.
        """
        latent_train = convert_index_points(tsne_train)[0]
        latent_val = convert_index_points(tsne_val)[0]
        latent_test = convert_index_points(tsne_test)[0]
        feature_ndims = convert_index_points(tsne_train)[1] 
        
        # Define the DFT-calculated values
        ytrain_dft = tf.constant(ytrain_dft, dtype=tf.float64)
        yval_dft =  tf.constant(yval_dft, dtype=tf.float64)
        ytest_dft = tf.constant(ytest_dft, dtype=tf.float64)
            
        if maxiters <= 0: 
            amp = tf.cast(amp, tf.float64)
            length_scale = tf.cast(length_scale, tf.float64)
            print("Prior on the amplitude of the kernel = %.4f" %amp.numpy())
            print("Prior on the width of the kernel = %.4f" %length_scale.numpy())
            logging.info("No bijector is applied to the priors ...")

            # Build the optimised kernel using the input hyperparameters
            Optkernel = tfk.MaternOneHalf(amp, 
                                          length_scale,
                                          feature_ndims=feature_ndims)
            gprm_dft = tfd.GaussianProcessRegressionModel(kernel=Optkernel,
                                                          index_points=latent_test,
                                                          observation_index_points=latent_train,
                                                          observations=ytrain_dft)
        else: 
            print("Requested optimisation with Adam algorithm at learning rate %s" %rate)
            print("Number of iterations = %s" %maxiters)
            print("Prior on the amplitude of the kernel = %s" %amp)
            print("Prior on the width of the kernel = %s" %length_scale)
            optimizer = tf.optimizers.Adam(learning_rate=rate)

            # Create trainable variables and apply positive constraint
            amp = tfp.util.TransformedVariable(initial_value=amp,
                                               bijector=tfb.Exp(),
                                               name="amp",
                                               dtype=tf.float64)
            length_scale = tfp.util.TransformedVariable(initial_value=length_scale,
                                                        bijector=tfb.Exp(),
                                                        name="length_scale",
                                                        dtype=tf.float64)
            
            def trainables():
                return [var.trainable_variables[0] for var in [amp, length_scale]]

            logging.info("Training GP on the training set to minimise MAE on the validation set ...")
            @tf.function
            def loss_fn():
                kernel = tfk.MaternOneHalf(amp, 
                                           length_scale,
                                           feature_ndims=feature_ndims)
                gp = tfd.GaussianProcess(kernel=kernel, index_points=latent_train)
                return -gp.log_prob(ytrain_dft)
            
            OptLoss = np.array([])
            OptAmp = np.array([])
            OptLength = np.array([])
            Optmae_val = np.array([])
            Optmse_val = np.array([])
            Optsae_val = np.array([])
            for i in tf.range(maxiters):
                with tf.GradientTape() as tape:
                    loss = loss_fn()
                grads = tape.gradient(loss, trainables())
                optimizer.apply_gradients(zip(grads, trainables()))
                OptLoss = np.append(OptLoss, loss.numpy()) 
                OptAmp = np.append(OptAmp, amp._value().numpy())
                OptLength = np.append(OptLength, length_scale._value().numpy())
                gprm = tfd.GaussianProcessRegressionModel(
                    kernel=tfk.MaternOneHalf(OptAmp[i],
                                             OptLength[i],
                                             feature_ndims=feature_ndims),
                    index_points=latent_val,
                    observation_index_points=latent_train,
                    observations=ytrain_dft)
                Optmae_val = np.append(Optmae_val, tf.losses.MAE(yval_dft, gprm.mean()).numpy())
                Optmse_val = np.append(Optmse_val, tf.losses.MSE(yval_dft, gprm.mean()).numpy())
                Optsae_val = np.append(Optsae_val, np.std(np.abs(gprm.mean().numpy() - yval_dft.numpy()))) 
                if i % 10 == 0 or i + 1 == maxiters:
                    print("At step %d: loss=%.4f, amplitude=%.4f, length_scale=%.4f, mae=%.4f, mse=%.4f, sae=%.4f, min(std)=%.4f, max(std)=%.4f"
                          %(i, OptLoss[i], OptAmp[i], OptLength[i], Optmae_val[i], Optmse_val[i], Optsae_val[i],
                            min(gprm.stddev().numpy()), max(gprm.stddev().numpy())))
            logging.info("Best-fitted parameters:")
            print("          amplitude: %.4f" %OptAmp[np.argmin(Optmae_val)])
            print("          length_scale: %.4f" %OptLength[np.argmin(Optmae_val)])

            logging.info("Building optimised kernel using the optimised hyperparameters ...")
            logging.info("GP predicting the test set ...")
            Optkernel = tfk.MaternOneHalf(OptAmp[np.argmin(Optmae_val)],
                                          OptLength[np.argmin(Optmae_val)],
                                          feature_ndims=feature_ndims)
            gprm_dft = tfd.GaussianProcessRegressionModel(kernel=Optkernel,
                                                          index_points=latent_test,
                                                          observation_index_points=latent_train,
                                                          observations=ytrain_dft)

        # Compute the Pearson correlation coefficient, MAE, MSE and
        # standard deviation on the absolute error (SAE) on the test set 
        mae_test =  tf.losses.MAE(ytest_dft.numpy(), gprm_dft.mean().numpy())
        mse_test = tf.losses.MSE(ytest_dft.numpy(), gprm_dft.mean().numpy())
        sae_test = tf.math.reduce_std(tf.abs(gprm_dft.mean().numpy() - ytest_dft.numpy()))
        R, p = pearsonr(x=ytest_dft.numpy(), y=gprm_dft.mean().numpy())
        print("Prediction: mae = %.4f, mse = %.4f, sae = %.4f, min(std) = %.4f, max(std) = %.4f, R = %.4f"
              %(mae_test,
                mse_test,
                sae_test,
                min(gprm_dft.stddev().numpy()),
                max(gprm_dft.stddev().numpy()),
                R))

        logging.info("Writing results to file ...")
        if maxiters > 0:
            np.save("%s/OptLoss.npy" %datadir, OptLoss)
            np.save("%s/OptAmp.npy" %datadir, OptAmp)
            np.save("%s/OptLength.npy" %datadir, OptLength)
            np.save("%s/Optmae_val.npy" %datadir, Optmae_val)
            np.save("%s/Optmse_val.npy" %datadir, Optmse_val)
            np.save("%s/Optsae_val.npy" %datadir, Optsae_val)
        np.save("%s/ytrain.npy" %datadir, ytrain_dft.numpy())
        np.save("%s/yval.npy" %datadir, yval_dft.numpy())
        np.save("%s/ytest.npy" %datadir, ytest_dft.numpy())
        np.save("%s/mae_test.npy" %datadir, mae_test)
        np.save("%s/mse_test.npy" %datadir, mse_test)
        np.save("%s/sae_test.npy" %datadir, sae_test)
        np.save("%s/gp_mean.npy" %datadir, gprm_dft.mean().numpy())
        np.save("%s/gp_stddev.npy" %datadir, gprm_dft.stddev().numpy())
        np.save("%s/gp_variance.npy" %datadir, gprm_dft.variance().numpy())

        if maxiters <= 0:
            return ( amp.numpy(),
                     length_scale.numpy(),
                     None,
                     None,
                     mae_test )
        else:
            return ( OptAmp[np.argmin(Optmae_val)],
                     OptLength[np.argmin(Optmae_val)],
                     np.min(Optmae_val),
                     Optmse_val[np.argmin(Optmae_val)],
                     mae_test )

        
    def active(datadir, prop, tsne_train, tsne_val, tsne_test, ytrain_dft,
               yval_dft, ytest_dft, maxiters, amp, length_scale, rate):
        """
        adam.active(datadir, prop, tsne_train, tsne_val, tsne_test, ytrain_dft, 
                    yval_dft, ytest_dft, maxiters, amp, length_scale, rate)

        A Gaussian Process (GP) with a Matern One Half kernel. The GP is first
        trained to minimise the MAE on the validation set. The best hyperparameters
        obtained during the GP training is used to build an optimised kernel 
        for predicting the test set.
    
        Inputs:
        datadir-        Directory into which results are written into.
        prop-           Optical property of interest.
        tsne_train-     Latent points for the training set.
        tsne_val-       Latent points for the validation set.
        tsne_test-      Latent points for the test set. 
        ytrain_dft-     DFT-calculated set for training.
        yval_dft-       DFT-calculated data for validation.
        ytest_dft-      DFT-calculated data for testing.
        maxiters-       Number of iterations for optimising 
                        hyperparameters.           
        amp-            Maximum value of the kernel.
        length_scale-   The width of the kernel.
        rate-           Learning rate for Adam optimisation.

        Outputs:
        1-            Optimised loss.
        2-            Optimised kernel amplitude.
        3-            Optimised kernel scale length.
        4-            Best kernel amplitude.
        5-            Best kernel length scale.
        6-            GP prediction.
        7-            Uncertainty on the GP prediction.
        8-            Variance on the GP prediction. 
        9-            Best MAE and its corresponding MSE, the 
                      standard deviation on the MAE on the 
                      test set.
        10-           Pearson correlation coefficient between the DFT-
                      calculated and GP-predicted optical property.
        """
        latent_train = convert_index_points(tsne_train)[0]
        latent_val = convert_index_points(tsne_val)[0]
        latent_test = convert_index_points(tsne_test)[0]
        feature_ndims = convert_index_points(tsne_train)[1]

        # Define the DFT-calculated values 
        ytrain_dft = tf.constant(ytrain_dft, dtype=tf.float64)        
        yval_dft = tf.constant(yval_dft, dtype=tf.float64)
        ytest_dft = tf.constant(ytest_dft, dtype=tf.float64)
        
        if maxiters <= 0: 
            amp = tf.cast(amp, tf.float64)
            length_scale = tf.cast(length_scale, tf.float64)
            print("Prior on the amplitude of the kernel = %.4f" %amp.numpy())
            print("Prior on the width of the kernel = %.4f" %length_scale.numpy())
            print("No bijector is applied to the priors ...")

            # Build the optimised kernel using the input hyperparameters
            Optkernel = tfk.MaternOneHalf(amp, 
                                          length_scale,
                                          feature_ndims=feature_ndims) 
            gprm_dft = tfd.GaussianProcessRegressionModel(kernel=Optkernel,
                                                          index_points=latent_test,
                                                          observation_index_points=latent_train,
                                                          observations=ytrain_dft)
        else:
            print("Requested optimisation with Adam algorithm at learning rate %s" %rate)
            print("Number of iterations = %s" %maxiters)
            print("Prior on the amplitude of the kernel = %s" %amp)
            print("Prior on the width of the kernel = %s" %length_scale)
            optimizer = tf.optimizers.Adam(learning_rate=rate)
            
            # Create a trainable variables and apply positive constraint
            amp = tfp.util.TransformedVariable(initial_value=amp,
                                               bijector=tfb.Exp(),
                                               name="amp",
                                               dtype=tf.float64)
            length_scale = tfp.util.TransformedVariable(initial_value=length_scale,
                                                        bijector=tfb.Exp(),
                                                        name="length_scale",
                                                        dtype=tf.float64)
            def trainables():
                return [var.trainable_variables[0] for var in [amp, length_scale]]

            logging.info("Training GP on the training set to minimise MAE on the validation set ...")            
            @tf.function
            def loss_fn():
                kernel = tfk.MaternOneHalf(amp, 
                                           length_scale,
                                           feature_ndims=feature_ndims) 
                gp = tfd.GaussianProcess(kernel=kernel, index_points=latent_train)
                return -gp.log_prob(ytrain_dft)

            OptLoss = np.array([])
            OptAmp = np.array([])
            OptLength = np.array([])
            Optmae_val = np.array([])
            Optmse_val = np.array([])
            Optsae_val = np.array([])
            for i in tf.range(maxiters):
                with tf.GradientTape() as tape:
                    loss = loss_fn()
                grads = tape.gradient(loss, trainables())
                optimizer.apply_gradients(zip(grads, trainables()))
                OptLoss = np.append(OptLoss, loss.numpy())
                OptAmp = np.append(OptAmp, amp._value().numpy())
                OptLength = np.append(OptLength, length_scale._value().numpy())
                gprm = tfd.GaussianProcessRegressionModel(
                    kernel=tfk.MaternOneHalf(OptAmp[i],
                                             OptLength[i],
                                             feature_ndims=feature_ndims),
                    index_points=latent_val,
                    observation_index_points=latent_train,
                    observations=ytrain_dft)
                Optmae_val = np.append(Optmae_val, tf.losses.MAE(yval_dft, gprm.mean().numpy()))
                Optmse_val = np.append(Optmse_val, tf.losses.MSE(yval_dft, gprm.mean().numpy()))
                Optsae_val = np.append(Optsae_val, np.std(np.abs(gprm.mean().numpy() - yval_dft.numpy())))
                if i % 10 == 0 or i + 1 == maxiters:
                    print("At step %d: loss=%.4f, amplitude=%.4f, length_scale=%.4f, mae=%.4f, mse=%.4f, sae=%.4f, min(std) = %.4f, max(std) = %.4f"
                          %(i, OptLoss[i], OptAmp[i], OptLength[i], Optmae_val[i], Optmse_val[i], Optsae_val[i],
                            min(gprm.stddev().numpy()),
                            max(gprm.stddev().numpy())))
            logging.info("Best-fitted parameters:")
            print("          amplitude: %.4f" %OptAmp[np.argmin(Optmae_val)])
            print("          length_scale: %.4f" %OptLength[np.argmin(Optmae_val)])
                
            logging.info("Building optimised kernel using the optimised hyperparameters ...")
            logging.info("GP predicting the test set ...") 
            Optkernel = tfk.MaternOneHalf(OptAmp[np.argmin(Optmae_val)],
                                          OptLength[np.argmin(Optmae_val)],
                                          feature_ndims=feature_ndims) 
            gprm_dft = tfd.GaussianProcessRegressionModel(kernel=Optkernel,
                                                          index_points=latent_test,
                                                          observation_index_points=latent_train,
                                                          observations=ytrain_dft)

        # Compute the Pearson correlation coefficient, MAE, MSE and
        # standard deviation on the absolute error (SAE) on the test set
        mae_test = tf.losses.MAE(ytest_dft.numpy(), gprm_dft.mean().numpy())
        mse_test = tf.losses.MSE(ytest_dft.numpy(), gprm_dft.mean().numpy())
        sae_test = tf.math.reduce_std(tf.abs(gprm_dft.mean().numpy() - ytest_dft.numpy()))
        R, p = pearsonr(x=ytest_dft.numpy(), y=gprm_dft.mean().numpy())
        print("Prediction: mae = %.4f, mse = %.4f, sae = %.4f, min(std) = %.4f, max(std) = %.4f, R = %.4f"
              %(mae_test,
                mse_test,
                sae_test,
                min(gprm_dft.stddev().numpy()),
                max(gprm_dft.stddev().numpy()),
                R))

        logging.info("Writing results to file ...")
        if maxiters > 0:
            np.save("%s/OptLoss.npy" %datadir, OptLoss)
            np.save("%s/OptAmp.npy" %datadir, OptAmp)
            np.save("%s/OptLength.npy" %datadir, OptLength)
            np.save("%s/Optmae_val.npy" %datadir, Optmae_val)
            np.save("%s/Optmse_val.npy" %datadir, Optmse_val)
            np.save("%s/Optsae_val.npy" %datadir, Optsae_val)
        np.save("%s/ytrain.npy" %datadir, ytrain_dft.numpy())
        np.save("%s/yval.npy" %datadir, yval_dft.numpy())
        np.save("%s/ytest.npy" %datadir, ytest_dft.numpy())        
        np.save("%s/gp_mean.npy" %datadir, gprm_dft.mean().numpy())
        np.save("%s/gp_stddev.npy" %datadir, gprm_dft.stddev().numpy())
        np.save("%s/gp_variance.npy" %datadir, gprm_dft.variance().numpy())
                    
        # Lets predict the test DFT values and estimate the
        # uncertainties on the prediction. Since a log-loss
        # was minimised, the variance is a better measure of
        # the uncertainty. For more information, go to 
        # https://www.kdnuggets.com/2018/10/introduction-active-learning.html
        if maxiters <= 0:
            return ( None,
                     None,
                     None,
                     amp,
                     length_scale, 
                     gprm_dft.mean().numpy(),
                     gprm_dft.stddev().numpy(),
                     gprm_dft.variance().numpy(),
                     None,
                     mae_test,
                     mse_test,
                     sae_test,
                     R )
        else:
            return ( OptLoss,
                     OptAmp,
                     OptLength, 
                     OptAmp[np.argmin(Optmae_val)],
                     OptLength[np.argmin(Optmae_val)],
                     gprm_dft.mean().numpy(),
                     gprm_dft.stddev().numpy(),
                     gprm_dft.variance().numpy(),
                     min(Optmae_val),
                     mae_test,
                     mse_test,
                     sae_test,
                     R )
