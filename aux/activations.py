"""
activations.py, SciML-SCD, RAL

Extracts the activations from the specified layer, scales 
these activations or apply tSNE. The output is then used 
as latent index points by the Gaussian process. 
"""
import sys 
import logging
import os 
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(levelname)s:gp-net: %(message)s")

import numpy as np
np.random.seed(1)
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt 

from tensorflow.compat.v2.keras import backend as K
from megnet.models import MEGNetModel


class latent:

    def train_test_split(datadir, prop, mlayer, activations_input_full, Xpool,
                         ytest, perp, ndims, niters):
        """
        latent.train_test_split(datadir, prop, mlayer, activations_input_full, 
                                Xpool, ytest, perp, ndims, niters)

        tSNE analysis or feature scaling of the activations of a mlayer of a 
        neural network.

        Inputs:
        datadir-                   Directory into which results are written into.
        prop-                      Optical property of interest.
        mlayer-                    Layer of a MEGNet model of interest. 
        activations_input_full-    Input to the specific mlayer for 
                                   extraction of activations for the full dataset. 
        Xpool-                     Structures in pool.
        ytest-                     Targets in the test set. 
        perp-                      Perplexity value for tSNE analysis.
        ndims-                     Dimensions of embedded space.
        niters-                    The maximum number of iterations for 
                                   tSNE optimisation.

        Outputs:
        1-                         GP latent points for the pool and test sets. 
        """
        if ndims == 1 or ndims > 3: 
            logging.error("Only ndims 0, 2 or 3 are allowed!")
            sys.exit() 
        model_pretrained = MEGNetModel.from_file("%s/fitted_%s_model.hdf5" %(datadir, prop))

        logging.info("Extracting activations from the %s layer ..." %mlayer)
        net_mlayer = [i.output for i in model_pretrained.mlayers if i.name.startswith("%s" %mlayer)]
        compute_graph = K.function([model_pretrained.input], [net_mlayer])
        extracted_activations_full = [ ]
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))

        if ndims > 0:
            logging.info("Dimensionality reduction using tSNE begins ...")
            print("Requested number of components = ", ndims)
            print("Using max iterations = ", niters)
            print("Processing perplexity = ", perp)
            from sklearn.manifold import TSNE            
            
            tsne_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                             perplexity=perp).fit_transform(np.squeeze(extracted_activations_full))
        elif ndims == 0:
            logging.info("Scaling each feature to range 0, 1 ...")
            from sklearn.preprocessing import MinMaxScaler
            
            tsne_full = MinMaxScaler().fit(np.squeeze(extracted_activations_full)).transform(
                np.squeeze(extracted_activations_full))

        tsne_pool = tsne_full[:len(Xpool)]
        tsne_test = tsne_full[len(Xpool):]

        logging.info("Writing results to file ...") 
        np.save(file="%s/latent_full.npy" %datadir, arr=tsne_full)
        np.save(file="%s/latent_pool.npy" %datadir, arr=tsne_pool)
        np.save(file="%s/latent_test.npy" %datadir, arr=tsne_test)

        if ndims > 0:
            logging.info("Saving tSNE plots ...")            
            if ndims == 2:
                plt.figure(figsize = [12, 6])
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(mlayer, niters, perp))
                plt.scatter(tsne_test[:,0], tsne_test[:,1], c=ytest)
            elif ndims == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize = [14, 6])
                ax = fig.add_subplot(111, projection="3d")
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(mlayer, niters, perp))
                ax.scatter(tsne_test[:,0], tsne_test[:,1], tsne_test[:,2], c=ytest)
            
            plt.savefig("%s/tSNE_%s.pdf" %(datadir, prop))
        
        return tsne_pool, tsne_test


    def k_fold(datadir, fold, prop, mlayer, activations_input_full, train_idx,
               val_idx, Xpool, perp, ndims, niters):
        """
        latent.k_fold(datadir, fold, prop, mlayer, activations_input_full, 
                      train_idx, val_idx, Xpool, perp, ndims, niters)
        
        tSNE analysis or feature scaling of the activations of a mlayer of a 
        neural network for k-fold cross-validation. 

        Inputs:
        datadir-                   Directory into which results are written into.
        fold-                      Number of fold to be processed.
        prop-                      Optical property of interest.  
        mlayer-                    Layer of a MEGNet model of interest. 
        activations_input_full-    Input to the specific layer for  extraction 
                                   of activations for the full dataset. 
        train_idx-                 Indices to extract training set from the pool.
        val_idx-                   Indices to extrct validation set from the pool.
        Xpool-                     Structures in pool.     
        perp-                      Perplexity value for tSNE analysis. 
        ndims-                     Dimensions of embedded space.  
        niters-                    The maximum number of iterations for tSNE 
                                   optimisation.
        
        Outputs:
        1-                         GP latent points for the training, validation,
                                   and test sets.
        """
        if ndims == 1 or ndims > 3:
            logging.error("Only ndims 2 or 3 is allowed!")
            sys.exit()
        model_pretrained = MEGNetModel.from_file("%s/fitted_%s_model.hdf5" %(datadir, prop))

        logging.info("Extracting activations from the %s layer ..." %mlayer)
        net_mlayer = [i.output for i in model_pretrained.mlayers if i.name.startswith("%s" %mlayer)]
        compute_graph = K.function([model_pretrained.input], [net_mlayer])
        extracted_activations_full = [ ]
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))

        if ndims > 0: 
            logging.info("Dimensionality reduction using tSNE begins ...")
            print("Requested number of components = ", ndims)
            print("Using max iterations = ", niters)
            print("Processing perplexity = ", perp)
            from sklearn.manifold import TSNE
            
            tsne_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                             perplexity=perp).fit_transform(np.squeeze(extracted_activations_full))
        elif ndims == 0:
            logging.info("Scaling each feature to range 0, 1 ...")
            from sklearn.preprocessing import MinMaxScaler

            tsne_full = MinMaxScaler().fit(np.squeeze(extracted_activations_full)).transform(
                np.squeeze(extracted_activations_full))
            
        logging.info("Writing results to file ...") 
        np.save(file="%s/latent_full.npy" %datadir, arr=tsne_full)
        tsne_pool = tsne_full[:len(Xpool)]
        np.save(file="%s/latent_pool.npy" %datadir, arr=tsne_pool)
        
        tsne_train = tsne_pool[train_idx]
        np.save(file="%s/latent_train.npy" %datadir, arr=tsne_train)
        
        tsne_val = tsne_pool[val_idx]
        np.save(file="%s/latent_val.npy" %datadir, arr=tsne_val)

        tsne_test  = tsne_full[len(Xpool):]
        np.save(file="%s/latent_test.npy" %datadir, arr=tsne_test)

        return tsne_train, tsne_val, tsne_test 
        
    
    def active(datadir, prop, mlayer, sampling, activations_input_full,
               Xfull, Xtest, ytest, Xtrain, Xval, perp, ndims, niters):
        """
        latent.active(datadir, prop, mlayer, sampling, activations_input_full, 
                      Xfull, Xtest, ytest, Xtrain, Xval, perp, ndims, niters)

        tSNE analysis or feature scalinf of the activations of a mlayer of a 
        neural network for active learning purposes. 

        Inputs:
        datadir-                  Directory into which results are written into. 
        prop-                     Optical property of interest.
        mlayer-                   Layer of a MEGNet model of interest. 
        sampling-                 Type of sampling the test set for performing 
                                  active learning. 
        activations_input_full-   Input to the specific layer for extraction
                                  of activations for the full dataset.
        Xfull-                    Structures of the full dataset. 
        Xtest-                    Structures in test set. 
        ytest-                    Targets in the test set.
        Xtrain-                   Structures for training. 
        Xval-                     Structures for validation. 
        perp-                     Perplexity value for tSNE analysis.
        ndims-                    Dimensions of embedded space.
        niters-                   The maximum number of iterations for tSNE
                                  optimisation. 

        Outputs:
        1-                         GP latent points for the full, pool, training, 
                                   validation, and test sets. 
        """
        if ndims == 1 or ndims > 3:
            logging.error("Only ndims 2 or 3 is allowed!")
            sys.exit() 
        model_pretrained = MEGNetModel.from_file("%s/fitted_%s_model.hdf5" %(datadir, prop))

        logging.info("Extracting activations from the %s layer ..." %mlayer)
        net_mlayer = [i.output for i in model_pretrained.mlayers if i.name.startswith("%s" %mlayer)]
        compute_graph = K.function([model_pretrained.input], [net_mlayer])
        extracted_activations_full = [ ]
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))

        if ndims > 0:
            logging.info("Dimensionality reduction using tSNE begins ...")
            print("Requested number of components = ", ndims)
            print("Using max iterations = ", niters)
            print("Processing perplexity = ", perp)
            from sklearn.manifold import TSNE
            
            tsne_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                             perplexity=perp).fit_transform(np.squeeze(extracted_activations_full))
        elif ndims == 0:
            logging.info("Scaling each feature to range 0, 1 ...")
            from sklearn.preprocessing import MinMaxScaler
            
            tsne_full = MinMaxScaler().fit(np.squeeze(extracted_activations_full)).transform(
                np.squeeze(extracted_activations_full))
            
        # Update the tsne values for the training and test sets 
        tsne_test = [ ]
        tsne_train = [ ]
        tsne_val = [ ]
        for test in Xtest:
            for full, tr in zip(Xfull, tsne_full):
                if test == full:
                    tsne_test.append(tr)
        for train in Xtrain:
            for full, tr in zip(Xfull, tsne_full):
                if train == full:
                    tsne_train.append(tr)
        for val in Xval:
            for full, tr in zip(Xfull, tsne_full):
                if val == full:
                    tsne_val.append(tr)

        tsne_full = np.array(tsne_full)
        tsne_train = np.array(tsne_train)
        tsne_val = np.array(tsne_val)
        tsne_test = np.array(tsne_test)

        print("\nWriting latent points to file ...")
        np.save(file="%s/latent_full.npy" %datadir, arr=tsne_full)
        np.save(file="%s/latent_train.npy" %datadir, arr=tsne_train)
        np.save(file="%s/latent_val.npy" %datadir, arr=tsne_val)
        np.save(file="%s/latent_test.npy" %datadir, arr=tsne_test)

        if ndims > 0: 
            logging.info("Saving tSNE plots ...")
            if ndims == 2:
                plt.figure(figsize = [12, 6])
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(mlayer, niters, perp))
                plt.scatter(tsne_test[:,0], tsne_test[:,1], c=ytest)
            elif ndims == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize = [14, 6])
                ax = fig.add_subplot(111, projection="3d")
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(mlayer, niters, perp))
                ax.scatter(tsne_test[:,0], tsne_test[:,1], tsne_test[:,2], c=ytest)

            if not os.path.isfile("%s/tSNE_%s.pdf" %(sampling, prop)):
                plt.savefig("%s/tSNE_%s.pdf" %(datadir, prop))

        return tsne_train, tsne_val, tsne_test

    
