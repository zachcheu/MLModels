'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from PIL import Image
from math import ceil

class NeuralNet:

    def __init__(self, layers, learningRate, regParam, epsilon=0.12, numEpochs=1000):
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
        self.epsilon = epsilon
        self.numEpochs = numEpochs
      

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

        one_hot = self.one_hot(y)

        for i,x in np.ndenumerate(self.layers):
            i = i[0]
            # +1 for bias
            self.theta[i] = np.random.random_sample((prev+1,x))
            self.theta[i] = (self.theta[i] - 0.5) * 2 * self.epsilon
            prev = x
        last = len(self.layers)
        self.theta[last] = np.random.random_sample((prev+1, 10))
        self.theta[last] = (self.theta[last] - 0.5) * 2 * self.epsilon
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

                error[last+1] = activation[last+1] - one_hot[i]
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
                self.theta[key] = self.theta[key] - (self.learningRate * delta[key])

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape
        output = []
        for i in range(n):
            x = X[i,:]
            predictions = self.forward(x)[len(self.theta)]
            output.append(predictions.argmax())
        return np.asarray(output)

    def one_hot(self, y):
        y = y.astype(int)
        n = y.shape[0]
        one_hot = np.zeros((n, 10))
        one_hot[np.arange(n), y] = 1
        return one_hot

    def forward(self, initial_activation):
        activation = {}
        activation[0] = initial_activation.reshape((1,initial_activation.shape[0]))
        for i in range(len(self.theta)):
            activation[i+1] = np.append(np.ones((1,1)), activation[i], axis=1) @ self.theta[i]
            activation[i+1] = 1/(1+np.exp(-activation[i+1]))
        return activation
    
    def visualizeHiddenNodes(self, filename):
        '''
        Outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        group_photo = Image.new('L', (106, 106))
        layer = 0
        minVal = np.min(self.theta[0][1:])
        maxVal = np.max(self.theta[0][1:])
        for i in range(5):
            for j in range(5):
                img = Image.new('L', (20, 20))
                pixelValues = self.theta[0][1:,layer].reshape((20,20))
                pixelValues -= minVal
                pixelValues *= 255/(maxVal-minVal)
                pixels = img.load()
                for x in range(img.size[0]):
                    for y in range(img.size[1]): 
                        pixels[x,y] = (ceil(pixelValues[x,y]))
                group_photo.paste(img, (1+(i*21), 1+(j*21)))
                layer +=1 
        group_photo.save(filename)
        