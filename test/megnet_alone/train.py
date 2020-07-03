from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import json

inputs = pd.read_pickle('./gaps_data.pkl')


boundary = int(len(inputs)/0.9*0.8)
epochs = 5
batch_size=56

Xtrain = inputs.iloc[0:boundary]['structure'] 
ytrain = inputs.iloc[0:boundary]['band_gap'] 

Xtest = inputs.iloc[boundary:]['structure'] 
ytest = inputs.iloc[boundary:]['band_gap'] 

nfeat_bond = 10
nfeat_global = 2
r_cutoff = 5
gaussian_centers = np.linspace(0, 5, 10)
gaussian_width = 0.5
distance_converter = GaussianDistance(gaussian_centers, gaussian_width)
#bond_convertor = CrystalGraph(bond_convertor=distance_convertor, cutoff=r_cutoff)
graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5))
graph_converter = CrystalGraph(bond_converter=distance_converter)
model = MEGNetModel(nfeat_bond, nfeat_global, 
                    graph_converter=graph_converter)

#model.from_file('fitted_gap_model.hdf5')

model.train(Xtrain, ytrain, epochs=epochs, batch_size=batch_size,
    validation_structures=Xtest, validation_targets=ytest,
    scrub_failed_structures=True)        
            
#model.save_model('fitted_gap_model.hdf5')

