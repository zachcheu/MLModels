'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
import numpy as np
import math



class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
    
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        
        regularize = regLambda * np.power(theta, 2).sum()
        n, d = X.shape
        costSum = 0
        for i in range(n):
            if y[i] == 0:
                costSum += -math.log(1-self.sigmoid(theta.dot(X[:i])))
            else:
                costSum += -math.log(self.sigmoid(theta.dot(X[:i])))
        return costSum + regularize

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d = X.shape
        gradient = np.empty(d)
        for j in range(0, d):
            update = self.sigmoid(X@theta.T) - y
            if j == 0:
                gradient[0] = update.sum()
            else:
                gradient[j] = np.multiply(update, X[:,j]).sum() + (regLambda) * theta[j]
        return gradient
        


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        ** the d here is different from above! (due to augmentation) **
        '''
        n = len(X)
        X_ = np.c_[np.ones((n,1)), X]
        n,d = X_.shape
        self.theta = np.random.normal(loc=0, scale=0.1, size=(d)).T
        iterCount = 0
        while iterCount < self.maxNumIters and not self.hasConverged(X_, y):
            iterCount += 1

    def hasConverged(self, X, y):
        prevTheta = np.copy(self.theta)
        gradient = self.computeGradient(self.theta, X, y, self.regLambda)
        converged = False
        self.theta = self.theta - self.alpha * gradient
        if np.sqrt(np.power(self.theta - prevTheta, 2).sum()) <= self.epsilon:
            converged = True
        return converged

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions, the output should be binary (use h_theta > .5)
        '''
        n = X.shape[0]
        X_ = np.c_[np.ones((n,1)), X]
        out = np.zeros((n,1))
        for i in range(n):
            if self.sigmoid(self.theta.dot(X_[i,:])) >= 0.5:
                out[i] = 1
            else:
                out[i] = 0
        return out

    def sigmoid(self, Z):
        '''
        Applies the sigmoid function on every element of Z
        Arguments:
            Z can be a (n,) vector or (n , m) matrix
        Returns:
            A vector/matrix, same shape with Z, that has the sigmoid function applied elementwise
        '''
        Z = 1/(1+np.exp(-Z))
        return Z
