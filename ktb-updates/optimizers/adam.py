"""
adam.py, SciML-SCD, RAL

Uses the adam optimiser to optimise the hyperparameters of the Gaussian 
Process. The optical properties of materials are predicted and their  
uncertainties are estimated. 
"""

import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions 
tfk = tfp.math.psd_kernels
tfb = tfp.bijectors 


class adam:

    def active(tsne_full, tsne_train, tsne_val, tsne_test, yfull_dft, ytrain_dft, yval_dft,
               ytest_dft, maxiters, amp, length_scale, rate):
        """
        We are using an exponentiated quadratic as the kernel for the 
        Gaussian Process (GP). This will result in a smooth prior on 
        functions sampled from the GP. 

        The adam algorithm is use to optimise the hyperparameters to 
        minimise the negative log likelihood. 
    
        Inputs:
        tsne_full-      Latent points for full dataset.
        tsne_train-     Latent points for the training set.
        tsne_val-       Latent points for the validation set.
        tsne_test-      Latent points for the test set. 
        yfull_dft-      Full DFT-calculated dataset.
        ytrain_dft-     DFT-calculated set for training.
        yval_dft-       DFT-calculated data for validation.
        ytest_dft-      DFT-calculated data for testing.
        maxiters-       Number of iterations for optimising 
                        hyperparameters.           
        amp-            Maximum value of the kernel.
        length_scale-   The width of the kernel.
        rate-           Learning rate for Adam optimisation.

        Outputs:
        1-              Uncertainties on the predicted optical 
                        property.
        """
        print("\nGaussian Process initiated ...")
        # Reshape the latent points so the kernel can take into
        # account the dimension of the latent points.
        if np.shape(tsne_full)[1] == 2:
            latent_full = tf.constant(tsne_full[..., np.newaxis], dtype=tf.float64)
            latent_train = tf.constant(tsne_train[..., np.newaxis], dtype=tf.float64)
            latent_val = tf.constant(tsne_val[..., np.newaxis], dtype=tf.float64)
            latent_test = tf.constant(tsne_test[..., np.newaxis], dtype=tf.float64)
        elif np.shape(tsne_full)[1] == 3:
            latent_full = tf.constant(tsne_full[..., np.newaxis, np.newaxis], dtype=tf.float64)
            latent_train = tf.constant(tsne_train[..., np.newaxis, np.newaxis], dtype=tf.float64)
            latent_val = tf.constant(tsne_val[..., np.newaxis, np.newaxis], dtype=tf.float64)
            latent_test = tf.constant(tsne_test[..., np.newaxis, np.newaxis], dtype=tf.float64)

        # Define the DFT-calculated values 
        yfull_dft = tf.constant(yfull_dft, dtype=tf.float64)
        ytrain_dft = tf.constant(ytrain_dft, dtype=tf.float64)        
        yval_dft = tf.constant(yval_dft, dtype=tf.float64)
        ytest_dft = tf.constant(ytest_dft, dtype=tf.float64)
        
        if maxiters == 0 or maxiters < 0:
            amp = tf.cast(amp, tf.float64)
            length_scale = tf.cast(length_scale, tf.float64)
            print("Prior on the amplitude of the kernel = %.4f" %amp.numpy())
            print("Prior on the width of the kernel = %.4f" %length_scale.numpy())
            print("No bijector is applied to the priors ...")

            # Build the optimised kernel using the input hyperparameters
            Optkernel = tfk.ExponentiatedQuadratic(amp, length_scale,
                                                   feature_ndims=np.shape(tsne_full)[1])
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

            print("\nTraining GP on the training set to minimise MAE on the validation set ...")            
            @tf.function
            def loss_fn():
                """ The loss function to be minimised for the training DFT values """
                kernel = tfk.ExponentiatedQuadratic(amp, length_scale,
                                                    feature_ndims=np.shape(tsne_full)[1])
                gp = tfd.GaussianProcess(kernel=kernel, index_points=latent_train)
                return -gp.log_prob(ytrain_dft)
            
            OptAmp = [ ]
            OptLength = [ ]
            mae_val = [ ]
            mse_val = [ ]
            for i in tf.range(maxiters):
                with tf.GradientTape() as tape:
                    loss = loss_fn()
                grads = tape.gradient(loss, trainables())
                optimizer.apply_gradients(zip(grads, trainables()))
                OptAmp.append(amp._value().numpy())
                OptLength.append(length_scale._value().numpy())
                gprm = tfd.GaussianProcessRegressionModel(
                    kernel=tfk.ExponentiatedQuadratic(amp._value().numpy(),
                                                      length_scale._value().numpy(),
                                                      feature_ndims=np.shape(tsne_full)[1]),
                    index_points=latent_val,
                    observation_index_points=latent_train,
                    observations=ytrain_dft)
                mae_val.append(tf.losses.MAE(yval_dft.numpy(), gprm.mean()))
                mse_val.append(tf.losses.MSE(yval_dft.numpy(), gprm.mean()))
                if i % 10 == 0 or i + 1 == maxiters:
                    print("At step %d: loss=%.4f, amplitude=%.4f, length_scale=%.4f, mae=%.4f, mse=%.4f"
                          %(i, loss, amp._value().numpy(), length_scale._value().numpy(),
                            mae_val[i], mse_val[i]))
            print("Best-fitted parameters:")
            print("amplitude: %.4f" %OptAmp[np.argmin(mae_val)])
            print("length_scale: %.4f" %OptLength[np.argmin(mae_val)])
                
            print("\nBuilding optimised kernel using the optimised hyperparameters ...")
            print("GP predicting the test set ...") 
            Optkernel = tfk.ExponentiatedQuadratic(OptAmp[np.argmin(mae_val)],
                                                   OptLength[np.argmin(mae_val)],
                                                   feature_ndims=np.shape(tsne_full)[1])
            gprm_dft = tfd.GaussianProcessRegressionModel(kernel=Optkernel,
                                                          index_points=latent_test,
                                                          observation_index_points=latent_train,
                                                          observations=ytrain_dft)
            print("Prediction: mae = %.4f, mse = %.4f" %(tf.losses.MAE(ytest_dft.numpy(), gprm_dft.mean().numpy()),
                                                         tf.losses.MSE(ytest_dft.numpy(), gprm_dft.mean().numpy())))

        # Lets predict the test DFT values and estimate the
        # uncertainties on the prediction. Since a log-loss
        # was minmised, the variance is a better measure of
        # the uncertainty. For more information, go to 
        # https://www.kdnuggets.com/2018/10/introduction-active-learning.html
        if maxiters > 0:
            return (gprm_dft.mean().numpy(),
                    gprm_dft.variance().numpy(),
                    min(mae_val),
                    tf.losses.MAE(ytest_dft.numpy(), gprm_dft.mean().numpy()),
                    OptAmp[np.argmin(mae_val)],
                    OptLength[np.argmin(mae_val)])
        else:
            return (gprm_dft.mean().numpy(),
                    gprm_dft.variance().numpy(),
                    None, 
                    tf.losses.MAE(ytest_dft.numpy(), gprm_dft.mean().numpy()),
                    amp,
                    length_scale)
