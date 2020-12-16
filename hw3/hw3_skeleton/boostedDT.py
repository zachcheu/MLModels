"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
"""

import numpy as np
from sklearn import tree


class BoostedDT:

    def __init__(self, num_boosting_iters=100, max_tree_depth=3):
        """
        Constructor
        """
        # TODO
        self.iters = num_boosting_iters
        self.depth = max_tree_depth
        self.tree_weight = []
        self.models = []

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        """
        # TODO: np.unique and np logical functions (logical_and/or/not) may be helpful to your implementation
        n,d = X.shape
        weight = np.zeros((n,))
        weight.fill(1/n)
        
        self.K = int(np.amax(y) - np.amin(y)) + 1
        
        for i in range(self.iters):
            clf = tree.DecisionTreeClassifier(max_depth=self.depth)
            clf.fit(X, y, sample_weight=weight)
            trainResults = clf.predict(X)
            # print(weight)
            # print("predict: ", trainResults)
            # print("actual: ", y)
            resultsInputNotEqual = np.not_equal(trainResults, y)
            incorrectIndex = np.argwhere(resultsInputNotEqual)
            # print(incorrectIndex)
            error = weight[incorrectIndex].sum()
            print(error)
            # error = (weight * resultsInputNotEqual).sum()
            B = 0.5 * (np.log((1-error)/error) + np.log(self.K-1))
            # print(resultsInputNotEqual)
            # weightChangeScale = np.where(resultsInputNotEqual, 1, np.exp(B))
            # print(weightChangeScale)
            # weight = np.multiply(weight, weightChangeScale)
            

            weight = weight * np.exp(B * resultsInputNotEqual)
            weight = np.multiply(weight, 1/weight.sum())
            # weight = weight/weight.sum()

            self.tree_weight.append(B)
            self.models.append(clf)

    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        """
        # TODO
        n,d = X.shape
        counter = np.zeros((n,self.K))
        for i, model in enumerate(self.models):
            classified = model.predict(X)
            for index, val in np.ndenumerate(classified):
                counter[index, int(val)] += self.tree_weight[i]
        return np.argmax(counter, axis=1)