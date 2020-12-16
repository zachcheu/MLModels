"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    
    return  np.power(X1.dot(X2.T) + 1, _polyDegree)# TODO


def myGaussianKernel(X1, X2, gamma = _gaussSigma):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    out = np.zeros((X1.shape[0], X2.shape[0]))
    den = 2 * gamma * gamma
    for i,x in enumerate(X1):
        for j,y in enumerate(X2):
            num = np.linalg.norm(x-y)**2
            out[i,j] = -(num/den)
    out = np.exp(out)

    return out# TODO

 