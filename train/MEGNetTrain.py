"""
MEGNetTrain.py, SciML-SCD, RAL

Trains on the optical properties of materials using the MEGNet 
of materials. Refer to https://github.com/materialsvirtuallab/megnet 
for more information on MEGNet. 
"""
import sys
import subprocess 
import logging 
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(levelname)s:gp-net: %(message)s")

import numpy as np
from keras.callbacks import ModelCheckpoint


class training:
 
    def train_test_split(datadir, prop, prev, model, batch, epochs, Xpool,
                         ypool, Xtest, ytest):
        """
        training.train_test_split(datadir, prop, prev, model, batch, epochs,
                                  Xpool, ypool, Xtest, ytest)
        
        MEGNet training on train-test split dataset. In this instance, the 
        pool is the training set. 

        Inputs:
        datadir-      Directory into which results are written into. 
        prop-         Optical property of interest.
        prev-         Best fitted MEGNet model.   
        model-        Featurised structures for training with MEGNet.
        batch-        Batch size for training.
        epochs-       Number of training iterations. 
        Xpool-        Structures for training. 
        ypool-        Targets for training. 
        Xtest-        Structures for testing.
        ytest-        targets for testing. 

        Outputs:
        1-            A fitted model of the optical property of interest.
        2-            Best model. Useable in the next round of training.
        """
        if not os.path.isdir(datadir):
            os.makedirs(datadir)
            
        logging.info("Writing data to file ...")
        np.save(file="%s/Xpool.npy" %datadir, arr=Xpool)
        np.save(file="%s/ypool.npy" %datadir, arr=ypool)
        np.save(file="%s/Xtest.npy" %datadir, arr=Xtest) 
        np.save(file="%s/ytest.npy" %datadir, arr=ytest) 
            
        if type(prev) == bool:
            if prev == False:
                logging.info("No previous model will be used ...")
                prev_file = None
            else: 
                logging.info("Searching for a previous model ...")
                prev_file = "%s/model-best-new-%s.h5" %(datadir, prop)
                if os.path.isfile(prev_file):
                    logging.info("Pre-trained model: %s found" %prev_file)
                    prev_file = prev_file
                else:
                    logging.info("No previous model found ...")
                    logging.info("Training without a previous model ...")
                    prev_file = None
        elif type(prev) == str:                    
            # For passing k-fold cross validation best fitted model 
            if os.path.isfile(prev):            
                logging.info("Pre-trained model: %s found" %prev)
                prev_file = prev
            else:
                logging.info("No previous model found ...")
                logging.info("Training without a previous model ...")
                prev_file = None 

        checkpoint = ModelCheckpoint("%s/model-best-new-%s.h5" %(datadir, prop),
                                     verbose=1, monitor="val_loss",
                                     save_best_only=True, mode="auto")
        model.train(Xpool, ypool, epochs=epochs, batch_size=batch,
                    validation_structures=Xtest, validation_targets=ytest,
                    scrub_failed_structures=True, prev_model=prev_file,
                    callbacks=[checkpoint])
        model.save_model("%s/fitted_%s_model.hdf5" %(datadir, prop))
        subprocess.call(["rm", "-r", "callback/"])


    def k_fold(datadir, fold, prop, prev, model, batch, epochs, Xtrain, ytrain,
               Xval, yval):
        """
        training.k_fold(fold, prop, prev, model, batch, epochs, Xtrain, ytrain
                        Xval, yval)

        MEGNet training on each fold of the k-fold cross-validation datasets.
        
        Inputs:
        datadir-    Directory into which results are written into.
        fold-       Number of fold to be processed.     
        prop-       Optical property of interest.
        prev-       Best fitted MEGNet model.
        model-      Featurised structures for training with MEGNet.
        batch-      Batch size for training.
        epochs-     Number of training iterations.
        Xtrain-     Structures for training.  
        ytrain-     Targets for training.   
        Xval-       Structures for validation. 
        yval-       Targets for validation. 

        Outputs:
        1-          A fitted model of the optical property of interest.
        2-          Best model. Useable in the next round of training.
        """
        if not os.path.isdir(datadir):
            os.makedirs(datadir) 

        logging.info("Writing data to file ...")
        np.save(file="%s/Xtrain.npy" %datadir, arr=Xtrain)
        np.save(file="%s/ytrain.npy" %datadir, arr=ytrain)
        np.save(file="%s/Xval.npy" %datadir, arr=Xval)
        np.save(file="%s/yval.npy" %datadir, arr=yval)

        if prev == False:
            logging.info("No previous model will be used ...")
            prev_file = None
        else:
            logging.info("Searching for a previous model ...")
            if fold == 0:
                prev_file = "%s/model-best-new-%s.h5" %(datadir, prop)
            else:
                prev_file = "k_fold/%s_results/0%s_fold/model-best-new-%s.h5" %(
                    prop, fold-1, prop)
            if os.path.isfile(prev_file):
                logging.info("Pre-trained model: %s found" %prev_file)
                prev_file = prev_file
            else:
                logging.info("No previous model found ...")
                logging.info("Training without a previous model ...")
                prev_file = None
                
        checkpoint = ModelCheckpoint("%s/model-best-new-%s.h5" %(datadir, prop),
                                     verbose=1, monitor="val_loss",
                                     save_best_only=True, mode="auto")
        model.train(Xtrain, ytrain, epochs=epochs, batch_size=batch,
                    validation_structures=Xval, validation_targets=yval,
                    scrub_failed_structures=True, prev_model=prev_file,
                    callbacks=[checkpoint])
        model.save_model("%s/fitted_%s_model.hdf5" %(datadir, prop))


    def active(datadir, i, prop, prev, model, sampling, batch, epochs, Xpool,
               ypool, Xtest, ytest):
        """
        training.active(datadir, i, prop, prev, model, sampling, batch, epochs, 
                        Xpool, ypool, Xtest, ytest)
        
        MEGNet training for active learning purposes. A pre-trained model
        in a previous query is used in the next query. 
    
        Inputs:
        datadir-            Directory into which results are written into.
        i-                  Number of active learning iterations
                            performed.
        prop-               Optical property of interest.
        prev-               Best fitted MEGNet model.
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
        2-                  Best model. Useable in the next round of 
                            training.
        """
        if not os.path.isdir(datadir): 
            os.makedirs(datadir)

        logging.info("Writing data to file ...")
        np.save(file="%s/Xpool.npy" %datadir, arr=Xpool) 
        np.save(file="%s/ypool.npy" %datadir, arr=ypool) 
        np.save(file="%s/Xtest.npy" %datadir, arr=Xtest) 
        np.save(file="%s/ytest.npy" %datadir, arr=ytest) 

        # For identifying location of best models to be used in the next iteration, i
        if i == 0:
            j = 0
        else:
            j = i - 1
            
        if prev == False:
            logging.info("No previous model will be used ...")
            prev_file = None
        else:
            logging.info("Searching for a previous model ...")
            prev_file = "%s/%s_results/0%s_model/model-best-new-%s.h5" %(
                sampling, prop, j, prop)
            if os.path.isfile(prev_file):
                logging.info("Pre-trained model: %s found" %prev_file)
                prev_file = prev_file
            else:
                logging.info("No previous model found ...")
                logging.info("Training without a previous model ...")
                prev_file = None 

        checkpoint = ModelCheckpoint("%s/model-best-new-%s.h5" %(datadir, prop), verbose=1,
                                     monitor="val_loss", save_best_only=True, mode="auto")
        model.train(Xpool, ypool, epochs=epochs, batch_size=batch, validation_structures=Xtest,
                    validation_targets=ytest, scrub_failed_structures=True, prev_model=prev_file,
                    callbacks=[checkpoint])
        model.save_model("%s/fitted_%s_model.hdf5" %(datadir, prop)) 
