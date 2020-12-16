'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in range(self.n_iter):
            
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta)
            # TODO: update equation here
            sumVals = []
            for j in range(d):
                sum = 0
                for k in range(n):
                    x = X[k,:]
                    sum += (np.dot(x, theta) - y[k]) * X[k,j]
                sumVals.append(sum.item(0),)
            theta = np.subtract(theta, self.alpha * (1/n) * np.array(sumVals).reshape((d,1)))
        # theta = np.linalg.inv(np.matrix.transpose(X)@X)@np.matrix.transpose(X)@y
        return theta
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        # TODO: add objective (cost) equation here
        n,d = X.shape
        inner = []
        #print(X)
        for i in range(n):
            x = X[i,:]
            inner.append(np.square(np.dot(x, theta) - y[i]))  
        return sum(inner)/(2*n)
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        #X = np.c_[np.ones((n,1)),X]
        n,d = X.shape
        #print(self.theta)
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # TODO:  add prediction function here
        n,d = X.shape
        out = []
        for i in n:
            x = X[i,:]
            out.append(np.dot(x, self.theta))
        return np.array(out)