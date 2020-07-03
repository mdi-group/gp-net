from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

inputs = pd.read_pickle('./band_gap_data.pkl')


boundary = int(len(inputs)*0.75)
epochs = 5
batch_size=56

Xtrain = inputs.iloc[0:boundary]['structure'] 
ytrain = inputs.iloc[0:boundary]['band_gap'] 

Xtest = inputs.iloc[boundary:]['structure'] 
ytest = inputs.iloc[boundary:]['band_gap'] 

for j in range(5):
    model = MEGNetModel.from_file('../entropy/0%s_model/fitted_band_gap_model.hdf5' % j)
    model.load_weights('../entropy/0%s_model/model-best-new-band_gap.h5' % j)
    preds = []
    vals = []
    for i in tqdm(range(len(Xtrain[-1000:]))):
        if ytrain[i] > 0:
            bg = model.predict_structure(Xtrain[i])
            preds.append(bg)
            vals.append(ytrain[i])
    print(mean_absolute_error(preds, vals))

