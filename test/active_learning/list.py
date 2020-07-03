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

split_pool = 0.8
active_cut = 0.05
start_cut = 0.5
queries = 5
prop = 'band_gap'
batch = 128
epochs = 1
ZeroVals = False
bond = 10
nfeat_global = 2
cutoff = 5
width = 0.5
fraction = (0.5, 0.5)
perp = 150
layer = 'readout_0'
ndims = 2
niters = 500
maxiters = 300
ampo = 7
length_scaleo = 7
rate = 0.001
query = 300
max_query = 5

'''
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
'''
Xpool = ['a', 'b', 'c', 'd']
Xunlab = ['e', 'f', 'g', 'h']
ypool = [1, 2, 3, 4]
yunlab = [5, 6, 1, 2]
for i in range(queries):
# Train MegNet
# Select the next round of structures
    query = 1
    max_query = 10
    dft_variance = yunlab
    EntropySelection = selection_fn.EntropySelection
    #mae_val_entropy.append(mae_val)
    #mae_gp_entropy.append(mae_gp)
    Xpool, ypool, Xunlab, yunlab = EntropySelection(i, Xpool, ypool, Xunlab, yunlab,
                             dft_variance, query, max_query)
    print('Pool', Xpool, ypool)
    print('Unlab', Xunlab, yunlab)
