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
import random

split_pool = 0.8
active_cut = 0.85
start_cut = 0.90
queries = 5
prop = 'band_gap'
batch = 128
epochs = 300 # Epochs of training megnet per cycle
### MegNet graph params
ZeroVals = False
bond = 10
nfeat_global = 2
cutoff = 5
width = 0.5
### MegNet graph params end
fraction = (0.5, 0.5)
perp = 150
layer = 'readout_0'
ndims = 2
niters = 500
maxiters = 300
ampo = 7
length_scaleo = 7
rate = 0.001 # Learning rate for GP
query = 50 # Number of samples to transfer per AL cycle
max_query = 15

model, Xfull, yfull, activations_input_full, Xpool, ypool, Xtest, ytest, val_frac \
              = megnet_input(prop, ZeroVals, bond, nfeat_global, cutoff, width, fraction)        

cut_a = int(len(Xfull)*active_cut)
print('Cutoff', cut_a)
activations_input_full = activations_input_full[:cut_a]
Xactive = Xfull[:cut_a]
yactive = yfull[:cut_a]
Xtest = Xfull[cut_a:]
ytest = yfull[cut_a:]
cut_p = int(len(Xactive)*start_cut)
Xpool = Xactive[:cut_p]
ypool = yactive[:cut_p]
Xunlab = Xactive[cut_p:]
yunlab = yactive[cut_p:]

for i in range(queries):
# Train MegNet
    print('Sample from y test for consistency', len(ytest), ytest[0])
    Xactive = np.concatenate((Xpool, Xunlab))
    yactive = np.concatenate((ypool, yunlab))
    training.active(i, prop, model, 'entropy', 
                    batch, epochs, Xpool[:cut_p], ypool[:cut_p], 
                    Xunlab[cut_p:], yunlab[cut_p:])

    randomlist = random.sample(range(0, query+1), query)
# Select the next round of structures
    EntropySelection = selection_fn.EntropySelection
    #mae_val_entropy.append(mae_val)
    #mae_gp_entropy.append(mae_gp)
    Xpool, ypool, Xunlab, yunlab = EntropySelection(i, Xpool, ypool, Xunlab, yunlab,
                             randomlist, query, max_query)
