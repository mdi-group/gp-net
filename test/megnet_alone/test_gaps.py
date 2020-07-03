from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import json

inputs = pd.read_pickle('./band_gap_data.pkl')


boundary = int(len(inputs)*0.75)
epochs = 5
batch_size=56

Xtrain = inputs.iloc[0:boundary]['structure'] 
ytrain = inputs.iloc[0:boundary]['band_gap'] 

Xtest = inputs.iloc[boundary:]['structure'] 
ytest = inputs.iloc[boundary:]['band_gap'] 

model_form = MEGNetModel.from_file('./fitted_gap_model.hdf5')

for i in range(10):
    bg = model.predict_structure(Xtrain[i])
    print(bg, ytrain[i])

