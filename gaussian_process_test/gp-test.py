# Imports
import os
import sys
import warnings
warnings.simplefilter("ignore")
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error

def build_gp(amplitude, length_scale):#, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale, feature_ndims=2)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_.reshape(-1, 1, 2))

# Use `tf.function` to trace the loss for more efficient evaluation.
@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(amplitude, length_scale):
  return gp_joint_model.log_prob({
      'amplitude': amplitude,
      'length_scale': length_scale,
      'observations': observations_
  })

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

data_limit = 5000
test_limit = 5500
len_est = 0.0102
amp_est = 2.6067
gp_train_steps = 30

with open('dense_0.pkl', 'rb') as f:
   data = pickle.load(f)
coords = data[0][:data_limit]
gaps = np.array(data[2][:data_limit])
coords = np.array([np.float64(a) for a in coords])
test_coords = data[0][data_limit:test_limit]
test_gaps = np.array(data[2][data_limit:test_limit])
test_coords = np.array([np.float64(a) for a in test_coords])

observations_ = gaps
observation_index_points_ = coords

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp
})

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=amp_est,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=len_est,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in
                       [amplitude_var,
                       length_scale_var]]

# Now we optimize the model parameters.
num_iters = gp_train_steps
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var, length_scale_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss
  if i % 10 == 0:
      print('Loss {0:8.4f} at step {1:5d}'.format(loss, i))
      print('Amplitude {0:8.4f} Length {1:8.4f}'.format(amplitude_var._value().numpy(), 
            length_scale_var._value().numpy()))
      print('--------------------------------------')

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))

## Calculate MAE on the trainig set
limit = 50
predictive_index_points_ = observation_index_points_[:limit].reshape(-1, 1, 2)
optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var, feature_ndims=2)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_[:500].reshape(-1, 1, 2),
    observations=observations_[:500])

num_samples = 10
samples = gprm.sample(num_samples)
estimates = []
for i in range(limit):
    sam = np.array([s[i] for s in samples])
    estimates.append([np.mean(sam), np.std(sam)])
est_mean = [e[0] for e in estimates]

print('MAE on training set: {0:8.3f} eV '.format(mean_absolute_error(est_mean, observations_[:limit])))

limit = 50
predictive_index_points_ = test_coords[:limit]
optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var, feature_ndims=2)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_.reshape(-1, 1, 2),
    observation_index_points=observation_index_points_[:5000].reshape(-1, 1, 2),
    observations=observations_[:5000])

num_samples = 50
samples = gprm.sample(num_samples)
estimates = []
for i in range(limit):
    sam = np.array([s[i] for s in samples])
    estimates.append([np.mean(sam), np.std(sam)])
est_mean = [e[0] for e in estimates]

print('MAE on test set: {0:8.3f} eV '.format(mean_absolute_error(est_mean, test_gaps[:limit])))

