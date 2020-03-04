from test_linreg_univariate import plotData1D
import numpy as np

filePath = "data/univariateData.dat"
file = open(filePath,'r')
allData = np.loadtxt(file, delimiter=',')
X = np.matrix(allData[:, :-1])
y = np.matrix((allData[:,-1])).T
n,d=X.shape

plotData1D(X,y)