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
active_cut = 0.75
start_cut = 0.0625
queries = 5
prop = 'band_gap'
batch = 256
epochs = 30 # Epochs of training megnet per cycle
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
ampo = 8
length_scaleo = 8
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
    print('Query ', i)
    print('Sample from y test for consistency', len(ytest), ytest[0])
    Xactive = np.concatenate((Xpool, Xunlab))
    yactive = np.concatenate((ypool, yunlab))
    training.active(i, prop, model, 'entropy', 
                    batch, epochs, Xpool, ypool, 
                    Xtest, ytest)

# Get the activations for the active set
    activations = []
    gaussian_centers = np.linspace(0, cutoff, bond)
    distance_converter = GaussianDistance(gaussian_centers, width)
    graph_converter = CrystalGraph(bond_converter=distance_converter)
    for s in Xactive:
        activations.append(StructureGraph.get_input(graph_converter, s))
# Obtain latent points
    tsne_active = latent.active(
       i, prop, perp, layer, 'entropy', activations, Xactive, Xpool, ypool,
       Xtest, val_frac, ndims, niters)
# Split the data
    tsne_pool = tsne_active[:len(ypool)]
    tsne_unlab = tsne_active[len(ypool):]
    cut = int(len(tsne_pool)*split_pool)
    tsne_train = tsne_pool[:cut]
    ytrain = ypool[:cut]
    tsne_val = tsne_pool[cut:]
    yval = ypool[cut:]

# Train the GP
    amp = ampo
    length_scale = length_scaleo
    gprm_dft, dft_variance, mae_val, mae_gp, amp, length_scale = adam.active(
       tsne_active, tsne_train, tsne_val, tsne_unlab, yactive, ytrain, yval, yunlab, 
       maxiters, amp, length_scale, rate)
# Select the next round of structures
    EntropySelection = selection_fn.EntropySelection
    #mae_val_entropy.append(mae_val)
    #mae_gp_entropy.append(mae_gp)
    Xpool, ypool, Xunlab, yunlab = EntropySelection(i, Xpool, ypool, Xunlab, yunlab,
                             dft_variance, query, max_query)
