import sys
sys.path.append('../../')
from aux.get_info import megnet_input
from train.MEGNetTrain import training 
from megnet.data.graph import GaussianDistance
from megnet.data.graph import StructureGraph
from megnet.data.crystal import CrystalGraph
from aux.activations import latent
from optimizers.adam import adam 
from aux.pool_sampling import selection_fn
import numpy as np

###################################################
queries = 20
prop = 'band_gap'
maxiters = 0 # iterations for the GP
ampo = 8
length_scaleo = 8
rate = 0.001 # Learning rate for GP
query = 50 # Number of samples to transfer per AL cycle
max_query = queries
data_directory = './starting_data/'
samp = 'random'
###################################################


tsne_train = np.load(data_directory + 'tsne_train.npy')
tsne_test = np.load(data_directory + 'tsne_test.npy')
tsne_val = np.load(data_directory + 'tsne_val.npy')
ytrain = np.load(data_directory + 'y_train.npy')
ytest = np.load(data_directory + 'y_test.npy')
yval = np.load(data_directory + 'y_val.npy')


for i in range(queries):
# 
    print('Query ', i)
    print('Sample from y test for consistency', len(ytest), ytest[0])

# Train the GP
    amp = ampo
    length_scale = length_scaleo
    gprm_dft, dft_variance, mae_val, mae_gp, amp, length_scale = adam.active(
                      tsne_train, tsne_val, tsne_test, ytrain, yval, ytest, 
                      maxiters, amp, length_scale, rate)
# Select the next round of structures
    if samp == "entropy":
        EntropySelection = selection_fn.EntropySelection                    
        tsne_pool, ypool, tsne_train, ytrain, tsne_test, ytest \
        = EntropySelection(i, tsne_train, ytrain, tsne_test, ytest,
                                       tsne_val, yval, dft_variance, query, max_query)
    elif samp == "random":
        RandomSelection = selection_fn.RandomSelection                    
        tsne_pool, ypool, tsne_train, ytrain, tsne_test, ytest \
        = RandomSelection(i, tsne_train, ytrain, tsne_test, ytest,
                           tsne_val, yval, dft_variance, query, max_query)
