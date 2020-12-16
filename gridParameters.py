import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from svmKernels import myGaussianKernel
from svmKernels import _gaussSigma

data = np.genfromtxt('data/svmTuningData.dat',
                     skip_header=1,
                     skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter=',')
print(np.array(data))
X = np.zeros((1,2))
# X2 = np.zeros((1,2))
Y = np.zeros((data.shape[0], 1))
for i,d in enumerate(data):
    temp = np.array([[d[0],d[1]]])
    X = np.concatenate((X, temp), axis=0)
    Y[i] = d[2]
    # if d[2] == -1:
    #     print(X1.shape, temp.shape)
        
    # else: 
    #     print(X2.shape, temp.shape)
    #     X2 = np.concatenate((X2, temp), axis=0)

# print(X1)
# print(X2)
clf = svm.SVC(kernel='rbf')
param_grid = {'C': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600, 1000],
'gamma': [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600, 1000] }
grid_search = GridSearchCV(clf, param_grid=param_grid)
X = X[1:,:]
grid_search.fit(X,Y)

print(grid_search)
print(grid_search.best_score_)
print(grid_search.best_estimator_.C)
print(grid_search.best_estimator_.gamma)

h = .02  # step size in the mesh
Y=Y.reshape((Y.shape[0],))
model = svm.SVC(C=grid_search.best_estimator_.C, kernel='rbf', gamma=grid_search.best_estimator_.gamma)
model.fit(X, Y)

plt.figure(figsize=(18, 6), dpi=100)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, predictions, cmap="Paired")
plt.scatter(X[:, 0], X[:, 1], c = Y, cmap="Paired", edgecolors="black")  # Plot the training points
plt.title('SVM with Equivalent Scikit_learn RBF Kernel for Comparison')
plt.axis('tight')
plt.show()
