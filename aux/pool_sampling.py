"""
pool_sampling.py, SciML-SCD, RAL

Samples data from the test set and transfers them 
into the training set for the purposes of active 
learning. Go to https://www.kdnuggets.com/2018/10/introduction-active-learning.html
for other means of sampling. 
"""

import numpy as np

class selection_fn:

    def EntropySelection(i, Xtrain, ytrain, Xtest, ytest, Xval, yval,
                         dft_variance, query, max_query):
        """
        selection_fn.EntropySelection(i, Xtrain, ytrain, Xtest, ytest,
                                      Xval, yval, dft_variance, query, 
                                      max_query)

        Sample selection based on the uncertainties obtained from the GP.

        Inputs:
        i-                  Number of active learning iterations
                            performed.
        Xtrain-             Structures for training.
        ytrain-             Targets for training 
        Xtest-              Structures in test test.
        ytest-              Targets in the test set.
        Xval-               Structures in pool.
        yval-               Targets in pool.
        dft_variance-       Variance on the GP predicted values.
        query-              Number of samples to move from the 
                            test set into the pool.
        max_query-          Maximum number of active learning 
                            iterations.

        Outputs
        1-                  Updated pool, test sets and their 
                            indices. 
        """
        idx = (np.argsort(dft_variance)[::-1])[:query]

        Xtrain = np.concatenate((Xtrain, Xtest[idx]))
        ytrain = np.concatenate((ytrain, ytest[idx]))
        
        Xtest = np.delete(Xtest, idx, axis=0)
        ytest = np.delete(ytest, idx, axis=0)

        Xpool = np.concatenate((Xtrain, Xval))
        ypool = np.concatenate((ytrain, yval))

        if i < max_query:
            print("\nEntropy sampling ..")
            print("Updated pool", ypool.shape)
            print("Updated training set", ytrain.shape)
            print("Updated test set:", ytest.shape)
            
        return idx, Xpool, ypool, Xtrain, ytrain, Xtest, ytest


    def RandomSelection(i, Xtrain, ytrain, Xtest, ytest, Xval, yval,
                        dft_variance, query, max_query):
        """
        selection_fn.RandomSelection(i, Xtrain, ytrain, Xtest, ytest, Xval, 
                                     yval, dft_variance, query, max_query) 

        A random selection of samples. The uncertainties  obtained from the 
        GP do not really matter.

        Inputs:
        i-                  Number of active learning iterations
                            performed.
        Xtrain-             Structures for training. 
        ytrain-             Targets for training.
        Xtest-              Structures in test set. 
        ytest-              Targets in test set. 
        Xval-               Structures in pool. 
        yval-               Targets in pool. 
        dft_variance-       Variance on the GP predicted values. 
        query-              Number of samples to move from the
                            test set into the pool.
        max_query-          Maximum number of active learning 
                            iterations.

        Outputs:
        1-                  Updated pool, test sets and their 
                            indices. 
        """
        np.random.seed(0)
        
        idx = np.sort(np.random.choice(len(dft_variance), query,
                                       replace=False))

        Xtrain = np.concatenate((Xtrain, Xtest[idx]))
        ytrain = np.concatenate((ytrain, ytest[idx]))

        Xtest = np.delete(Xtest, idx, axis=0)
        ytest = np.delete(ytest, idx, axis=0)
                
        Xpool = np.concatenate((Xtrain, Xval)) 
        ypool = np.concatenate((ytrain, yval))

        if i < max_query:
            print("\nRandom sampling ...")
            print("Updated pool:", ypool.shape)
            print("Updated training set", ytrain.shape)
            print("Updated test set:", ytest.shape)
        
        return idx, Xpool, ypool, Xtrain, ytrain, Xtest, ytest
