from nn import NeuralNet
import numpy as np
from sklearn.metrics import accuracy_score

trainX = np.genfromtxt('data/digitsX.dat', delimiter=',')
trainY = np.genfromtxt('data/digitsY.dat', delimiter=',')
layers = np.array([25])
clf = NeuralNet(layers, 1.0, 0)
clf.fit(trainX, trainY)
accuracy = accuracy_score(trainY, clf.predict(trainX))
print(accuracy)