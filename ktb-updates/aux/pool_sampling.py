"""
pool_sampling.py, SciML-SCD, RAL

Samples data from the test set and transfers them 
into the training set for the purposes of active 
learning. Go to https://www.kdnuggets.com/2018/10/introduction-active-learning.html
for other means of sampling. 
"""

import numpy as np

class selection_fn:

    def EntropySelection(i, Xpool, ypool, Xtest, ytest, dft_variance,
                         query, max_query):
        """
        selection_fn.EntropySelection(i, Xpool, ypool, Xtest, ytest, 
                                      dft_variance, query, max_query)

        Sample selection based on the uncertainties obtained from the GP.

        Inputs:
        i-                  Number of active learning iterations
                            performed.
        Xpool-              Structures in pool.
        ypool-              Targets in pool.
        Xtest-              Structures in test test.
        ytest-              Targets in the test set.
        dft_variance-       Variance on the GP predicted values.
        query-              Number of samples to move from the 
                            test set into the pool.
        max_query-          Maximum number of active learning 
                            iterations.

        Outputs
        1-                  Updated pool and test sets.
        """
        print('Number in test set {}, number in pool {}'.format(
              len(Xtest), len(Xpool)))
        idx = (np.argsort(dft_variance)[::-1])[:query]
        Xtest = np.array(Xtest)
        Xpool = np.array(Xpool)
        ytest = np.array(ytest)
        ypool = np.array(ypool)
        Xpool = np.concatenate((Xtest[idx], Xpool))
        ypool = np.concatenate((ytest[idx], ypool))

        Xtest = np.delete(Xtest, idx)
        ytest = np.delete(ytest, idx)

        if i < max_query:
            print("\nEntropy sampling ..")
            print("Updated pool", ypool.shape)
            print("Updated test set:", ytest.shape)
            
        return Xpool, ypool, Xtest, ytest


    def RandomSelection(i, Xpool, ypool, Xtest, ytest, dft_variance,
                        query, max_query):
        """
        selection_fn.RandomSelection(i, Xpool, ypool, Xtest, ytest,
                                     dft_variance, query, max_query)

        A random selection of samples. The uncertainties 
        obtained from the GP do not really matter.

        Inputs:
        i-                  Number of active learning iterations
                            performed.
        Xpool-              Structures in pool. 
        ypool-              Targets in pool. 
        Xtest-              Structures in test set. 
        ytest-              Targets in test set. 
        dft_variance-       Variance on the GP predicted values. 
        query-              Number of samples to move from the
                            test set into the pool.
        max_query-          Maximum number of active learning 
                            iterations.

        Outputs:
        1-                  Updated pool and test sets.
        """
        np.random.seed(0)        
        idx = np.sort(np.random.choice(len(dft_variance), query,
                                       replace=False))

        Xpool = np.concatenate((Xpool, Xtest[idx]))
        ypool = np.concatenate((ypool, ytest[idx]))

        Xtest = np.delete(Xtest, idx)
        ytest = np.delete(ytest, idx) 

        if i < max_query:
            print("\nRandom sampling ...")
            print("Updated pool:", ypool.shape)
            print("Updated test set:", ytest.shape)

        return Xpool, ypool, Xtest, ytest
