"""
activations.py, SciML-SCD, RAL

Extracts the activations from the specified layer, reduces
their dimensions and the transformed values are used as 
latent index points by the Gaussian process. 
"""

import sys 
import numpy as np
from sklearn.manifold import TSNE

from tensorflow.compat.v2.keras import backend as K
from megnet.models import MEGNetModel


class latent:
    
    def active(i, prop, perp, layer, sampling, activations_input_full, Xfull, Xpool,
               ypool, Xtest, val_frac, ndims, niters):
        """
        latent.active(i, prop, perp, layer, sampling, activations_input_full, Xfull, 
                      Xpool, Xtest, val_frac, ndims, niters)

        tSNE analysis of the activations of a layer for active learning purposes. 

        Inputs:
        i-                         Number of active learning iterations  
                                   performed.
        prop-                      Optical property of interest.
        perp-                      Perplexity value for tSNE analysis.
        layer-                     Neural network layer of interest. 
        sampling-                  Type of sampling the test set for performing 
                                   active learning. 
        activations_input_full-    Input to the specific layer for 
                                   extraction of activations for the full dataset.
        Xfull-                     Structures of the full dataset. 
        Xpool-                     Structures in pool.   
        Xtest-                     Structures in test test. 
        val_frac-                  Fraction of the training set for validation
                                   in active learning. 
        ndims-                     Dimensions of tSNE embedded space.
        niters-                    The maximum number of iterations for 
                                   optimisation with tSNE.

        Outputs:
        1-                         GP latent points for the full, pool, training, 
                                   validation, and test sets. 
        """
        if ndims < 2 or ndims > 3:
            sys.exit("Only ndims 2 or 3 is allowed!")
        model_pretrained = MEGNetModel.from_file("%s/0%s_model/fitted_%s_model.hdf5" %(sampling, i, prop))

        print("Extracting activations from the %s layer ..." %layer)
        net_layer = [i.output for i in model_pretrained.layers if i.name.startswith("%s" %layer)]
        compute_graph = K.function([model_pretrained.input], [net_layer])
        extracted_activations_full = []
        for full in activations_input_full:
            extracted_activations_full.append(compute_graph(full))
        
        print("\nDimensionality reduction using tSNE begins ...")
        print("Requested number of components = ", ndims)
        print("Using max iterations = ", niters)
        print("Processing perplexity = ", perp)
        tsne_full = TSNE(n_components=ndims, n_iter=niters, n_jobs=-1, random_state=0,
                         perplexity=perp).fit_transform(np.squeeze(extracted_activations_full))
        tsne_pool_idx = [ ]
        tsne_test_idx = [ ]
        for Xp in Xpool:
            for num, (Xf, tr) in enumerate(zip(Xfull, tsne_full)):
                if Xp == Xf:
                    tsne_pool_idx.append(num)
        for Xt in Xtest:
            for num, (Xf, tr) in enumerate(zip(Xfull, tsne_full)):
                if Xt == Xf:
                    tsne_test_idx.append(num)

        tsne_pool = tsne_full[tsne_pool_idx]
        tsne_test = tsne_full[tsne_test_idx]
        val_boundary = int(val_frac * len(tsne_pool))
        tsne_val = tsne_pool[-val_boundary:]
        tsne_train = tsne_pool[:-val_boundary]

        yval = ypool[-val_boundary:]        
        ytrain = ypool[:-val_boundary]

#        print("Validation set:", yval.shape)

        #return tsne_full, tsne_pool, tsne_train, tsne_val, tsne_test, ytrain, yval
        return tsne_full
