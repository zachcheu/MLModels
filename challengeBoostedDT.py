from boostedDT import BoostedDT
from bestClassifier import BestClassifier
import numpy as np
from sklearn.metrics import accuracy_score

trainData = np.genfromtxt('data/challengeTrainLabeled.dat',
                     delimiter=',')

testData = np.genfromtxt('data/challengeTestUnlabeled.dat',
                     delimiter=',')

X = trainData[:, :-1]
y = trainData[:, -1]

n, d = X.shape
nTrain = int(0.5*n)  # training on 50% of the data

# split the data
Xtrain = X[:nTrain, :]
ytrain = y[:nTrain]
Xtest = X[nTrain:, :]
ytest = y[nTrain:]

# train the boosted DT
# model = BoostedDT()
model = BestClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

accuracy= accuracy_score(ytest, ypred)
print("Model accuracy = " + str(accuracy))
np.savetxt('data/predictions-BestClassifier.dat', model.predict(testData), delimiter=',')