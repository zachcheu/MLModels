'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from PIL import Image
from math import ceil

class NeuralNet:

    def __init__(self, layers, learningRate, regParam, momentum, epsilon=0.12, numEpochs=1000):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	learningRate - the learning rate for backpropagation
        	regParam - the regularization parameter for the cost function
        	epsilon - one half the interval around zero for setting the initial weights
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.learningRate = learningRate
        self.regParam = regParam
        self.momentum = momentum
        self.epsilon = epsilon
        self.numEpochs = numEpochs
        self.previous = None
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
        prev = d
        self.theta = {}

        for i,x in np.ndenumerate(self.layers):
            i = i[0]
            # +1 for bias
            self.theta[i] = np.random.random_sample((prev+1,x))
            self.theta[i].fill(0.1)
            prev = x
        last = len(self.layers)
        self.theta[last] = np.random.random_sample((prev+1, 1))
        self.theta[last].fill(0.1)
        #theta[0] = layer between input and first layer
        #theta[last] = layer between l-1 to output layer

        for _ in range(self.numEpochs):
            delta = {}
            for i in range(n):
                activation = self.forward(X[i,:])

                #compute error layer by layer, backwards
                error = {}
                # theta = (nprev, ncurr)
                # error[j+1] = (1,ncurr)
                # error[j] = (1,nprev)
                error[last+1] = 2 * (activation[last+1]-y[i]) * activation[last+1] * (1-activation[last+1])
                for j in range(last, 0, -1):
                    # print((activation[j] * (1-activation[j])).shape)
                    # gprime = (np.append(np.ones((1,1)), activation[j], axis=1) * (1-np.append(np.ones((1,1)), activation[j], axis=1)))
                    gprime = activation[j] * (1- activation[j])
                    error[j] = (error[j+1] @ self.theta[j].T)[:,1:] * gprime
                    
                # compute gradients
                for j in range(0, last+1):
                    # print(activation[j].shape)
                    # print(error[j+1].shape)
                    if j not in delta:
                        delta[j] = np.append(np.ones((1,1)), activation[j], axis=1).T @ error[j+1]
                    else:
                        delta[j] += np.append(np.ones((1,1)), activation[j], axis=1).T @ error[j+1]
            
            # average gradients
            for key in delta.keys():
                delta[key] = delta[key]/n
                regularize = self.theta[key] * self.regParam
                regularize[0, :] = 0
                delta[key] += regularize
            
            # change weights
            for key in delta.keys():
                if self.previous is None:
                    self.theta[key] = self.theta[key] - (self.learningRate * delta[key])
                else:
                    self.theta[key] = self.theta[key] - (self.learningRate * delta[key]) - self.momentum * self.previous[key]
            
            print("Weights: ", self.theta[0][1], self.theta[0][2], self.theta[0][0], self.theta[1][1], self.theta[0][0])
            print("Gradients: ", delta[0][1], delta[0][2], delta[0][0], delta[1][1], delta[1][0])
            self.previous = delta

    def forward(self, initial_activation):
        activation = {}
        activation[0] = initial_activation.reshape((1,initial_activation.shape[0]))
        for i in range(len(self.theta)):
            activation[i+1] = np.append(np.ones((1,1)), activation[i], axis=1) @ self.theta[i]
            activation[i+1] = 1/(1+np.exp(-activation[i+1]))
        return activation

if __name__ == "__main__":
    layers = np.array([1])
    clf = NeuralNet(layers, 0.3, 0, 0.9, 0, 2)
    clf.fit(np.array([[1,0],[0,1]]), np.array([1,0]))
