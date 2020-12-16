# from sklearn.model_selectio import KFold
# from sklearn import svm
# import numpy as np
# from sklearn.metrics import accuracy_score 
# results = []
# trainData = np.genfromtxt('data/challengeTrainLabeled.dat',
#                      delimiter=',')

# testData = np.genfromtxt('data/challengeTestUnlabeled.dat',
#                      delimiter=',')
# XTrain = trainData[:, :-1]
# yTrain = trainData[:, -1]
# for c in [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600]:
#     for g in [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600]:
#         clf = svm.SVC(C=c, gamma =g ,kernel="rbf")
#         fold = KFold(n_splits = 10)
#         accuracy = 0
#         for train, test in fold.split(XTrain):
#             clf.fit(XTrain[train], yTrain[train])
#             prediction = clf.predict(XTrain[test])
#             accuracy += accuracy_score(yTrain[test], prediction)
#         accuracy /= 10
#         print("accuracy: ", accuracy, " c: ", c, " gamma: ", g)



import sklearn
class BestClassifier():
    def __init__(self):
        self.clf = sklearn.svm.SVC(C=6, gamma=0.1 ,kernel="rbf")
    
    def fit(self, X, y):
        self.clf.fit(X,y)
    
    def predict(self, X):
        return self.clf.predict(X)