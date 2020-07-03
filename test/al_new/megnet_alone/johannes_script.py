import sys
sys.path.append('../../')
from aux.get_info import megnet_input
from train.MEGNetTrain import training

model, Xfull, yfull, activations_input_full, Xpool, ypool, Xtest, ytest, val_frac \
   = megnet_input('band_gap', ZeroVals=False, bond=10, 
   nfeat_global=2, cutoff=5, width=0.5, fraction=0.9)

training.active(0, 'band_gap', model, 'entropy',
                batch, 5, Xpool, ypool,
                Xtest, ytest)
