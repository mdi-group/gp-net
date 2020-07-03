"""
MEGNetTrain.py, SciML-SCD, RAL

Trains on the optical properties of materials using the MEGNet 
of materials. Refer to https://github.com/materialsvirtuallab/megnet 
for more information on MEGNet. 
"""

import os
import sys
import numpy as np
from keras.callbacks import ModelCheckpoint


class training:

    def active(i, prop, model, sampling, batch, epochs, Xpool, ypool, Xtest, ytest, restart=False):
        """
        training.active(i, prop, model, sampling, batch, epochs, Xpool, ypool, Xtest, 
                        ytest)
        
        MEGNet training for active learning purposes. A pre-trained model
        in a previous query is used in the next query. 
    
        Inputs:
        i-                  Number of active learning iterations
                            performed.
        prop-               Optical property of interest.
        model-              Featurised structures for training with 
                            MEGNet.         
        sampling-           Type of sampling for transfer of data
                            from the test set to the pool. 
        batch-              Batch size for training. 
        epochs-             Number of training iterations. 
        Xpool-              Structures for training.
        ypool-              Targets for training. 
        Xtest-              Structures for testing.
        ytest-              Targets for testing. 

        Outputs:
        1-                  A fitted model of the optical property of 
                            interest. 
        """
        if not os.path.isdir("%s/0%s_model/" %(sampling, i)):
            os.makedirs("%s/0%s_model/" %(sampling, i))

        # For identifying location of best models to be used in the next iteration, i
        if i == 0:
            j = 0
        else:
            j = i - 1
            
        print("\nSearching for a pre-trained model ...")
        prev_file = "%s/0%s_model/model-best-new-%s.h5" %(sampling, j, prop)
        if os.path.isfile(prev_file and restart):
            print("Pre-trained model: %s found" %prev_file)
            prev_file = prev_file
        else:
            print("No pre-trained model found ...")
            prev_file = None
        checkpoint = ModelCheckpoint("%s/0%s_model/model-best-new-%s.h5" %(sampling, i, prop),
                                     verbose=1, monitor="val_loss", save_best_only=True,
                                     mode="auto")
        model.train(Xpool, ypool, epochs=epochs, batch_size=batch, validation_structures=Xtest,
                    validation_targets=ytest, scrub_failed_structures=True, prev_model=prev_file,
                    callbacks=[checkpoint])
        model.save_model("%s/0%s_model/fitted_%s_model.hdf5" %(sampling, i, prop))
