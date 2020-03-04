"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
"""

import numpy as np


class NaiveBayes:

    def __init__(self, use_laplace_smoothing=True):
        """
        Constructor
        """
        # TODO
        self.laplace = use_laplace_smoothing


    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        """
        # TODO: np.unique and np logical functions (logical_and/or/not) may be helpful to your implementation
        n,d = X.shape
        self.K = int(np.amax(y) - np.amin(y)) + 1
        self.classProb = {}
        for yval in np.nditer(y):
            yval = int(yval)
            if yval not in self.classProb:
                self.classProb[yval] = 0
            self.classProb[yval] += 1/n

        # featureClassProbability = np.zeros((n,d))
        if self.laplace:
            self.attributeClassificationTracker = np.ones((d,self.K))
        else:
            self.attributeClassificationTracker = np.zeros((d,self.K))
        for index, val in np.ndenumerate(X):
            if val > 0:
                rowIndex = index[0]
                featureIndex = index[1]
                classification = y[rowIndex]
                self.attributeClassificationTracker[featureIndex, classification] += val
        # self.attributeClassificationTracker = np.log(np.multiply(attributeClassificationTracker, 1/np.sum(attributeClassificationTracker, axis=1).reshape(d,1)))

    
    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        """
        # TODO
        return np.argmax(self.predictProbs(X),axis=1)

    def predictProbs(self, X):
        """
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        """
        # TODO
        n,d = X.shape
        classPredictions = np.zeros((n, self.K))
        for i in range(n):
            row = X[i,:]
            for j in range(self.K):
                probabilityFromOccurance = np.multiply(self.attributeClassificationTracker[:,j], 1/np.sum(self.attributeClassificationTracker[:,j]))
                logOfProbability = np.log(probabilityFromOccurance)
                multiplyLogOfProbabilityForOccurances = np.multiply(logOfProbability, row)
                classPredictions[i,j] = np.log(self.classProb[j]) + np.sum(multiplyLogOfProbabilityForOccurances)
        classPredictions = np.subtract(classPredictions, np.amax(classPredictions, axis=1).reshape(n,1))
        classPredictions = np.exp(classPredictions)
        classPredictions = np.multiply(classPredictions, 1/np.sum(classPredictions, axis=1).reshape(n,1))
        return classPredictions


class OnlineNaiveBayes:
    # You can use the class definition line below to subclass if you wish
    # class OnlineNaiveBayes(NaiveBayes):

    def __init__(self, use_laplace_smoothing=True):
        """
        Constructor
        """
        # TODO
        self.laplace = use_laplace_smoothing
        self.classProb = {}
        self.count = 0
        self.classCount = {}
        self.indexMatching = {}
        self.reverseIndexMatching = {}
        # self.featureMatching = {}
        self.reverseFeatureMatching = {}
        self.nextClassIndex = 0
        # self.nextFeatureIndex = 0
        self.attributeClassificationTracker = None

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        """
        # TODO: np.unique and np logical functions (logical_and/or/not) may be helpful to your implementation
        n,d = X.shape
        
        for yval in np.nditer(y):
            yval = int(yval)
            if yval not in self.classCount:
                self.classCount[yval] = 0
            self.classCount[yval] += 1
            self.count += 1

        for key in self.classCount.keys():
            self.classProb[key] = self.classCount[key]/self.count
        
        for index, val in np.ndenumerate(X):
            rowIndex = index[0]
            featureIndex = index[1]
            classification = y[rowIndex]
            if self.attributeClassificationTracker is None:
                self.indexMatching[classification] = self.nextClassIndex
                self.reverseIndexMatching[self.nextClassIndex] = classification
                self.nextClassIndex += 1
                if self.laplace:
                    self.attributeClassificationTracker = np.ones((d,1))
                else:
                    self.attributeClassificationTracker = np.zeros((d,1))
            else:
                if classification not in self.indexMatching:
                    self.indexMatching[classification] = self.nextClassIndex
                    self.reverseIndexMatching[self.nextClassIndex] = classification

                    self.nextClassIndex += 1
                    if self.laplace:
                        self.attributeClassificationTracker= np.hstack([self.attributeClassificationTracker, np.ones((d, 1))])
                    else:
                        self.attributeClassificationTracker = np.hstack([self.attributeClassificationTracker, np.zeros((d,1))])
            classIndex = self.indexMatching[classification]
            self.attributeClassificationTracker[featureIndex, classIndex] += val
        # print(self.indexMatching)
        # print(y)

    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        """
        # TODO
        return np.argmax(self.predictProbs(X),axis=1)

    def predictProbs(self, X):
        """
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        """
        # TODO
        n,d = X.shape
        dim, K = self.attributeClassificationTracker.shape
        classPredictions = np.zeros((n, K))
        for i in range(n):
            row = X[i,:]
            for j in range(K):
                probabilityFromOccurance = np.multiply(self.attributeClassificationTracker[:,j], 1/np.sum(self.attributeClassificationTracker[:,j]))
                logOfProbability = np.log(probabilityFromOccurance)
                multiplyLogOfProbabilityForOccurances = np.multiply(logOfProbability, row)
                classPredictions[i,j] = np.log(self.classProb[j]) + np.sum(multiplyLogOfProbabilityForOccurances)
        classPredictions = np.subtract(classPredictions, np.amax(classPredictions, axis=1).reshape(n,1))
        classPredictions = np.exp(classPredictions)

        order = [None] * K
        for i in range(K):
            order[i] = self.indexMatching[i]
        print(order)
        classPredictions = np.multiply(classPredictions, 1/np.sum(classPredictions, axis=1).reshape(n,1))
        return classPredictions[:,order]
