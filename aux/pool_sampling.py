"""
Samples data from the test set and transfers them
into the training set for the purposes of active
learning. Go to https://www.kdnuggets.com/2018/10/introduction-active-learning.html
for other means of sampling.
"""

import logging
import numpy

log = logging.getLogger("gp-net")

numpy.random.seed(0)


class SelectionFunction:
    """

    :param Xtrain: Structures for training.
    :param ytrain: Targets for training
    :param Xtest: Structures in test set.
    :param ytest: Targets in the test set.
    :param dft_variance: Variance on the GP predicted values.
    :param max_query: Maximum number of active learning iterations.
    """

    def __init__(self, Xtrain, ytrain, Xtest, ytest, dft_variance, query, max_query):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self.dft_variance = dft_variance
        self.query = query
        self.max_query = max_query

    def entropy_based(self, i, Xval, yval, query):
        """
        Sample selection based on the uncertainties obtained from the GP.

        :param i: Number of active learning iterations performed.
        :param Xval: Structures in pool.
        :param yval: Targets in pool.
        :param query: Number of samples to move from the test set into the pool.

        :return: Updated pool, test sets and their indices.
        """
        idx = (numpy.argsort(self.dft_variance)[::-1])[:query]

        Xtrain = numpy.concatenate((self.Xtrain, self.Xtest[idx]))
        ytrain = numpy.concatenate((self.ytrain, self.ytest[idx]))

        Xtest = numpy.delete(self.Xtest, idx, axis=0)
        ytest = numpy.delete(self.ytest, idx, axis=0)

        Xpool = numpy.concatenate((Xtrain, Xval))
        ypool = numpy.concatenate((ytrain, yval))

        if i < self.max_query:
            print("\nEntropy sampling ..")
            print("Updated pool", ypool.shape)
            print("Updated training set", ytrain.shape)
            print("Updated test set:", ytest.shape)

        return idx, Xpool, ypool, Xtrain, ytrain, Xtest, ytest

    def random_based(
        self,
        i,
        Xval,
        yval,
        query,
    ):
        """
        A random selection of samples. The uncertainties  obtained from the
        GP do not really matter.

        Inputs:
        :param i: Number of active learning iterations performed.
        :param Xval: Structures in pool.
        :param yval: Targets in pool.

        :param query: Number of samples to move from the test set into the pool.

        :return: Updated pool, test sets and their indices.
        """
        idx = numpy.sort(
            numpy.random.choice(len(self.dft_variance), query, replace=False)
        )
        Xtrain = numpy.concatenate((self.Xtrain, self.Xtest[idx]))
        ytrain = numpy.concatenate((self.ytrain, self.ytest[idx]))

        Xtest = numpy.delete(self.Xtest, idx, axis=0)
        ytest = numpy.delete(self.ytest, idx, axis=0)

        Xpool = numpy.concatenate((Xtrain, Xval))
        ypool = numpy.concatenate((ytrain, yval))

        if i < self.max_query:
            print("\nRandom sampling ...")
            print("Updated pool:", ypool.shape)
            print("Updated training set", ytrain.shape)
            print("Updated test set:", ytest.shape)

        return idx, Xpool, ypool, Xtrain, ytrain, Xtest, ytest
