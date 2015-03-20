from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from numpy import newaxis as na
import matplotlib.pyplot as plt
import abc
import copy
from warnings import warn

# Added by Enrique Noriega
from scipy.stats import norm

from pyhsmm.basic.abstractions import Distribution, GibbsSampling



class Probit(GibbsSampling, Distribution):
    ''' Probit model for the UA Context model of the REACH team

        The cumulative Gaussian distribution has zero mean and identity covariance matrix

        Author: Enrique Noriega
        Email: enoriega@email.arizona.edu
    '''

    def __init__(self, W, l):
        ''' W is the weight matrix and l is the state vector

            l is a binary vector and W a real matrix
        '''

        self.W = np.matrix(W)
        self.l = np.matrix(l)

        # Enforce l to be a column vector
        if l.shape[1] > 1:
            l = l.T

        assert W.shape[1] == l.shape[0], "The weight matrix and the state vector should be equivalent"
        assert ((l <= 1) & (l >= 0)).all(), "l should be a binary vector"

        # Compute the weights vector
        self.w = W*l #These should be numpy's matrix objects so this works

    @np.vectorize
    def _threshold(s, p):
        ''' Internal method to check wether an element is above a threshold or not '''
        return 0. if s>= p else 1.

    def rvs(self, size=1):
        ''' Generates a random variate (sample)

            It samples from a bernoulli with parameters as self.w
        '''

        if type(size) in (list, tuple):
            size = size[0] if len(size) > 0 else 1

        variate = np.matrix(np.zeros((size, self.w.shape[0])))

        for j in xrange(self.w.shape[0]):
            variate[:, j] = self._threshold(norm.rvs(size=(size, 1)), self.w[j])

        return variate


    def log_likelihood(self, X):
        ''' Computes the log likelihood according to the following formula:

        p(\mathbf{y} | \mathbf{w}) = \log \left[ \prod_{k} \Phi(w_k)^{y_k} (1 - \Phi(w_k))^{1-y_k} \right]

        where \mathbf{w} is the product \mathbf{W} \times \mathbf{l}
        '''

        X = X.astype(np.bool)

        if X.shape[1] < X.shape[0]:
            X = X.T

        # for i in xrange(X.shape[0]):
        #     x = X[i,:]
        #
        #     # Now compute the joint log probability
        #     if x.shape[1] > 1:
        #         x = x.T
        #
        #     ret[i] = np.log(norm.cdf(self.w[x == 1])).sum() + np.log(np.ones([1, x.shape[0] - x.sum()]) - norm.cdf(self.w[x == 0])).sum()
        #
        # return ret



        w = np.tile(self.w, X.shape[1])

        im = np.multiply(w, X)
        t1 = np.log(norm.cdf(im)).sum(axis=0)

        iX = np.invert(X)
        im = np.ones(w.shape) - norm.cdf(np.multiply(w, iX))
        t2 = np.log(im).sum(axis=0)

        return t1 + t2

    # TODO: Implement this to actually do something
    def resample(self,data=[]):
        pass
