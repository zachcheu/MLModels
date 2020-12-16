'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.regLambda = reg_lambda
        self.theta = None
        self.degree= degree

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        #TODO
        out = np.copy(X)
        for i in range(2, degree):
            out = np.concatenate((out, np.power(X, i)), axis=1)
        return out


    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        #TODO
        n = len(X)
        X = self.polyfeatures(X, self.degree + 1)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        X = (X - self.mean) / self.std
        
        X_ = np.c_[np.ones((n, 1)), X]

        reg_matrix = self.regLambda * np.eye(self.degree+1)
        reg_matrix[0, 0] = 0
        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # TODO
        # print(X)
        n = len(X)
        X = self.polyfeatures(X, self.degree+1)
        X = (X - self.mean) / self.std
        X_ = np.c_[np.ones((n, 1)), X]
        
        # print(self.theta.T)
        return X_.dot(self.theta)


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------

def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    
    for i in range(1, n):
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(Xtrain[:i+1], Ytrain[:i+1])
        errorTest[i] = pow(model.predict(Xtest) - Ytest, 2)
        errorTrain[i] = np.power(model.predict(Xtrain[:i+1]) - Ytrain[:i+1], 2).mean()

    return errorTrain, errorTest
