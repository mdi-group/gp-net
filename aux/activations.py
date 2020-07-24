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

    def train_test_split(datadir, prop, layer, activations_input_full, Xpool,
                         ytest, perp, ndims, niters):
        """
        latent.train_test_split(datadir, prop, layer, activations_input_full, 
                                Xpool, ytest, perp, ndims, niters)

        tSNE analysis or feature scaling of the activations of a layer of a 
        neural network.

        Inputs:
        datadir-                   Directory into which results are written into.
        prop-                      Optical property of interest.
        layer-                     Layer of a MEGNet model of interest. 
        activations_input_full-    Input to the specific layer for 
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
        if ndims > 3: 
            logging.error("0 <= ndims < 4!") 
            sys.exit() 
        model_pretrained = MEGNetModel.from_file("%s/fitted_%s_model.hdf5" %(datadir, prop))

        logging.info("Extracting activations from the %s layer ..." %layer)
        net_layer = [i.output for i in model_pretrained.layers if i.name.startswith("%s" %layer)]
        compute_graph = K.function([model_pretrained.input], [net_layer])
        extracted_activations_full = [ ]
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))

        activations = np.array(np.squeeze(extracted_activations_full))
        if ndims in (0, 1):
            if np.ndim(activations) > 2:
                logging.error("Dimension of extracted activations > 2 so apply tSNE instead!")
                sys.exit()
            if ndims == 0:
                logging.info("No pre-processing on the extracted activations ...")
                latent_full  = activations
            elif ndims == 1:
                logging.info("Scaling each feature to range 0, 1 ...")
                from sklearn.preprocessing import MinMaxScaler
            
                latent_full = MinMaxScaler().fit(activations).transform(activations)
        elif ndims > 1:
            logging.info("Dimensionality reduction using tSNE begins ...")
            print("Requested number of components = ", ndims)
            print("Using max iterations = ", niters)
            print("Processing perplexity = ", perp)
            from sklearn.manifold import TSNE            
            
            latent_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                          perplexity=perp).fit_transform(activations)

        latent_pool = latent_full[:len(Xpool)]
        latent_test = latent_full[len(Xpool):]

        logging.info("Writing results to file ...") 
        np.save(file="%s/latent_full.npy" %datadir, arr=latent_full)
        np.save(file="%s/latent_pool.npy" %datadir, arr=latent_pool)
        np.save(file="%s/latent_test.npy" %datadir, arr=latent_test)

        if ndims == 0:
            logging.info("Saving extracted activations plot ...")
            plt.figure(figsize = [12, 6])
            plt.title("Raw activations from %s layer" %layer)
            plt.scatter(latent_test[:,0], latent_test[:,1], c=ytest)
            plt.savefig("%s/activations_%s.pdf" %(datadir, prop))
        elif ndims == 1:
            logging.info("Saving scaled activations plot ...")
            plt.figure(figsize = [12, 6])
            plt.title("Scaled activations from %s layer" %layer)
            plt.scatter(latent_test[:,0], latent_test[:,1], c=ytest)
            plt.savefig("%s/activations_%s.pdf" %(datadir, prop))
        elif ndims > 1:
            logging.info("Saving tSNE plots ...")            
            if ndims == 2:
                plt.figure(figsize = [12, 6])
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(layer, niters, perp))
                plt.scatter(latent_test[:,0], latent_test[:,1], c=ytest)
            elif ndims == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize = [14, 6])
                ax = fig.add_subplot(111, projection="3d")
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(layer, niters, perp))
                ax.scatter(latent_test[:,0], latent_test[:,1], latent_test[:,2], c=ytest)
            
            plt.savefig("%s/tSNE_%s.pdf" %(datadir, prop))
        
        return latent_pool, latent_test


    def k_fold(datadir, fold, prop, layer, activations_input_full, train_idx,
               val_idx, Xpool, perp, ndims, niters):
        """
        latent.k_fold(datadir, fold, prop, layer, activations_input_full, 
                      train_idx, val_idx, Xpool, perp, ndims, niters)
        
        tSNE analysis or feature scaling of the activations of a layer of a 
        neural network for k-fold cross-validation. 

        Inputs:
        datadir-                   Directory into which results are written into.
        fold-                      Number of fold to be processed.
        prop-                      Optical property of interest.  
        layer-                     Layer of a MEGNet model of interest. 
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
        if ndims > 3:
            logging.error("0 <= ndims < 4 !")
            sys.exit()
        model_pretrained = MEGNetModel.from_file("%s/fitted_%s_model.hdf5" %(datadir, prop))

        logging.info("Extracting activations from the %s layer ..." %layer)
        net_layer = [i.output for i in model_pretrained.layers if i.name.startswith("%s" %layer)]
        compute_graph = K.function([model_pretrained.input], [net_layer])
        extracted_activations_full = [ ]
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))

        activations = np.array(np.squeeze(extracted_activations_full))
        if ndims in (0, 1):
            if np.ndim(activations) > 2:
                logging.error("Dimension of extracted activations > 2 so apply tSNE instead!")
                sys.exit()
            if ndims == 0:
                logging.info("No pre-processing on the extracted activations ...")
                latent_full  = activations
            elif ndims == 1:
                logging.info("Scaling each feature to range 0, 1 ...")
                from sklearn.preprocessing import MinMaxScaler

                latent_full = MinMaxScaler().fit(activations).transform(activations)
        elif ndims > 1:
            logging.info("Dimensionality reduction using tSNE begins ...")
            print("Requested number of components = ", ndims)
            print("Using max iterations = ", niters)
            print("Processing perplexity = ", perp)
            from sklearn.manifold import TSNE
            
            latent_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                               perplexity=perp).fit_transform(activations)
            
        logging.info("Writing results to file ...") 
        np.save(file="%s/latent_full.npy" %datadir, arr=latent_full)
        
        latent_pool = latent_full[:len(Xpool)]
        np.save(file="%s/latent_pool.npy" %datadir, arr=latent_pool)
        
        latent_train = latent_pool[train_idx]
        np.save(file="%s/latent_train.npy" %datadir, arr=latent_train)
        
        latent_val = latent_pool[val_idx]
        np.save(file="%s/latent_val.npy" %datadir, arr=latent_val)

        latent_test  = latent_full[len(Xpool):]
        np.save(file="%s/latent_test.npy" %datadir, arr=latent_test)

        return latent_train, latent_val, latent_test 
        
    
    def active(datadir, prop, layer, sampling, activations_input_full,
               Xfull, Xtest, ytest, Xtrain, Xval, perp, ndims, niters):
        """
        latent.active(datadir, prop, layer, sampling, activations_input_full, 
                      Xfull, Xtest, ytest, Xtrain, Xval, perp, ndims, niters)

        tSNE analysis or feature scaling of the activations of a layer of a 
        neural network for active learning purposes. 

        Inputs:
        datadir-                  Directory into which results are written into. 
        prop-                     Optical property of interest.
        layer-                    Layer of a MEGNet model of interest. 
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
        if ndims > 3:
            logging.error("0 <= ndims < 4!")
            sys.exit() 
        model_pretrained = MEGNetModel.from_file("%s/fitted_%s_model.hdf5" %(datadir, prop))

        logging.info("Extracting activations from the %s layer ..." %layer)
        net_layer = [i.output for i in model_pretrained.layers if i.name.startswith("%s" %layer)]
        compute_graph = K.function([model_pretrained.input], [net_layer])
        extracted_activations_full = [ ]
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))

        activations = np.array(np.squeeze(extracted_activations_full))
        if ndims in (0, 1):
            if np.ndim(activations) > 2:
                logging.error("Dimension of extracted activations > 2 so apply tSNE instead!")
                sys.exit()
            if ndims == 0:
                logging.info("No pre-processing on the extracted activations ...")
                latent_full  = activations
            elif ndims == 1:
                logging.info("Scaling each feature to range 0, 1 ...")
                from sklearn.preprocessing import MinMaxScaler

                latent_full = MinMaxScaler().fit(activations).transform(activations)
        elif ndims > 1:
            logging.info("Dimensionality reduction using tSNE begins ...")
            print("Requested number of components = ", ndims)
            print("Using max iterations = ", niters)
            print("Processing perplexity = ", perp)
            from sklearn.manifold import TSNE
            
            latent_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                             perplexity=perp).fit_transform(activations)
            
        # Update the tsne values for the training and test sets
        latent_test = [ ]
        latent_train = [ ]
        latent_val = [ ]
        for test in Xtest:
            for full, tr in zip(Xfull, latent_full):
                if test == full:
                    latent_test.append(tr)
        for train in Xtrain:
            for full, tr in zip(Xfull, latent_full):
                if train == full:
                    latent_train.append(tr)
        for val in Xval:
            for full, tr in zip(Xfull, latent_full):
                if val == full:
                    latent_val.append(tr)

        latent_train = np.array(latent_train)
        latent_val = np.array(latent_val)
        latent_test = np.array(latent_test)

        print("\nWriting latent points to file ...")
        np.save(file="%s/latent_full.npy" %datadir, arr=latent_full)
        np.save(file="%s/latent_train.npy" %datadir, arr=latent_train)
        np.save(file="%s/latent_val.npy" %datadir, arr=latent_val)
        np.save(file="%s/latent_test.npy" %datadir, arr=latent_test)

        if ndims == 0:
            logging.info("Saving extracted activations plot ...")
            plt.figure(figsize = [12, 6])
            plt.title("Raw activations from %s layer" %layer)
            plt.scatter(latent_test[:,0], latent_test[:,1], c=ytest)
            plt.savefig("%s/activations_%s.pdf" %(datadir, prop))
        elif ndims == 1:
            logging.info("Saving scaled activations plot ...")
            plt.figure(figsize = [12, 6])
            plt.title("Scaled activations from %s layer" %layer)
            plt.scatter(latent_test[:,0], latent_test[:,1], c=ytest)
            plt.savefig("%s/activations_%s.pdf" %(datadir, prop))        
        elif ndims > 1: 
            logging.info("Saving tSNE plots ...")
            if ndims == 2:
                plt.figure(figsize = [12, 6])
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(layer, niters, perp))
                plt.scatter(latent_test[:,0], latent_test[:,1], c=ytest)
            elif ndims == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize = [14, 6])
                ax = fig.add_subplot(111, projection="3d")
                plt.title("tSNE transformed activations of %s layer \nNumber of iterations = %s \nperplexity = %s"
                          %(layer, niters, perp))
                ax.scatter(latent_test[:,0], latent_test[:,1], latent_test[:,2], c=ytest)

            if not os.path.isfile("%s/tSNE_%s.pdf" %(sampling, prop)):
                plt.savefig("%s/tSNE_%s.pdf" %(datadir, prop))

        return latent_train, latent_val, latent_test

    
