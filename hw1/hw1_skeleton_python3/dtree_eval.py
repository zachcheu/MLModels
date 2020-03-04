'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

def removePercentage(data, index):
    n, d = data.shape
    groups = np.arange(0, n, math.floor((n-1)/10))
    count = n-groups[-1]
    for i in range(count):
        groups[-i-1] += count-i
    #print(groups)
    #print(data[:groups[index+1],].shape)
    return data[:groups[index+1],]

def resultScore(clf, Xtrain, ytrain, Xtest, ytest):
    clf = clf.fit(Xtrain,ytrain)
    y_pred = clf.predict(Xtest)
    return accuracy_score(ytest, y_pred)

def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    n,d = data.shape

    fileX = data[:, 1:]
    filey = np.array([data[:, 0]]).T
    n,d = fileX.shape
    meanGraph= [[],[],[],[],[]]
    stdGraph = [[],[],[],[],[]]
    
    for j in range(10):
        meanDecisionTreeAccuracy = 0
        stddevDecisionTreeAccuracy = 0
        meanDecisionStumpAccuracy = 0
        stddevDecisionStumpAccuracy = 0
        meanDT3Accuracy = 0
        stddevDT3Accuracy = 0

        meanDT5Accuracy = 0
        stddevDT5Accuracy = 0

        meanDT7Accuracy = 0
        stddevDT7Accuracy = 0

        # split the data
        groups = np.arange(0, n, math.floor((n-1)/10))
        count = n-groups[-1]
        for i in range(count):
            groups[-i-1] += count-i

        for k in range(100):
            X = np.copy(fileX)
            y = np.copy(filey)
            # shuffle the data
            idx = np.arange(n)
            np.random.seed(k)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            dtsum = 0
            dssum = 0
            d3sum = 0
            d5sum = 0
            d7sum = 0
            dtvalues = []
            dsvalues = []
            d3values = []
            d5values = []
            d7values = []
            for i in range(10):
                Xtest = X[groups[i]:groups[i+1]]
                ytest = y[groups[i]:groups[i+1]]
                XtrainTotal = np.delete(X, np.arange(groups[i], groups[i+1]), axis=0)
                ytrainTotal = np.delete(y, np.arange(groups[i], groups[i+1]), axis=0)
                
                Xtrain = removePercentage(XtrainTotal, j)
                # print(XtrainTotal.shape, Xtrain.shape, j)
                ytrain = removePercentage(ytrainTotal, j)

                #decision tree
                score = resultScore(tree.DecisionTreeClassifier(), Xtrain, ytrain, Xtest, ytest)
                dtsum += score
                dtvalues.append(score)

                #decision stump
                score = resultScore(tree.DecisionTreeClassifier(max_depth=1), Xtrain, ytrain, Xtest, ytest)
                dssum += score
                dsvalues.append(score)

                #dt3
                score = resultScore(tree.DecisionTreeClassifier(max_depth=3), Xtrain, ytrain, Xtest, ytest)
                d3sum += score
                d3values.append(score)

                #5
                score = resultScore(tree.DecisionTreeClassifier(max_depth=5), Xtrain, ytrain, Xtest, ytest)
                d5sum += score
                d5values.append(score)

                #7
                score = resultScore(tree.DecisionTreeClassifier(max_depth=7), Xtrain, ytrain, Xtest, ytest)
                d7sum += score
                d7values.append(score)

            meanDecisionTreeAccuracy += dtsum/10.0
            stddevDecisionTreeAccuracy += np.std(dtvalues)

            meanDecisionStumpAccuracy += dssum/10
            stddevDecisionStumpAccuracy += np.std(dsvalues)

            meanDT3Accuracy += d3sum/10
            stddevDT3Accuracy += np.std(d3values)

            meanDT5Accuracy += d5sum/10
            stddevDT5Accuracy += np.std(d5values)

            meanDT7Accuracy += d7sum/10
            stddevDT7Accuracy += np.std(d7values)
        
        meanDecisionTreeAccuracy /= 100.0
        stddevDecisionTreeAccuracy /= 100.0

        meanDecisionStumpAccuracy /= 100
        stddevDecisionStumpAccuracy /= 100

        meanDT3Accuracy /= 100
        stddevDT3Accuracy /= 100

        meanDT5Accuracy /= 100
        stddevDT5Accuracy /= 100

        meanDT7Accuracy /= 100
        stddevDT7Accuracy /= 100

        meanGraph[0].append(meanDecisionTreeAccuracy)
        meanGraph[1].append(meanDecisionStumpAccuracy)
        meanGraph[2].append(meanDT3Accuracy)
        meanGraph[3].append(meanDT5Accuracy)
        meanGraph[4].append(meanDT7Accuracy)

        stdGraph[0].append(stddevDecisionTreeAccuracy)
        stdGraph[1].append(stddevDecisionStumpAccuracy)
        stdGraph[2].append(stddevDT3Accuracy)
        stdGraph[3].append(stddevDT5Accuracy)
        stdGraph[4].append(stddevDT7Accuracy)

    top = [[],[],[],[],[]]
    bottom=[[],[],[],[],[]]
    for i in range(10):
        for j in range(5):
            top[j].append(meanGraph[j][i] + stdGraph[j][i])
            bottom[j].append(meanGraph[j][i] - stdGraph[j][i])
        
    plt.plot(meanGraph[0], color='green', label='Full Decision Tree')
    plt.plot(meanGraph[1], color='red', label='Decision Stump Tree')
    plt.plot(meanGraph[2], color='blue', label='Max-Depth 3 Tree')
    plt.plot(meanGraph[3], color='purple', label='Max-Depth 4 Tree')
    plt.plot(meanGraph[4], color='yellow', label='Max-Depth 5 Tree')

    plt.fill_between(np.arange(0,10), top[0], bottom[0], color = 'green', alpha=0.1)
    plt.fill_between(np.arange(0,10), top[1], bottom[1], color = 'red', alpha=0.1)
    plt.fill_between(np.arange(0,10), top[2], bottom[2], color = 'blue', alpha=0.1)
    plt.fill_between(np.arange(0,10), top[3], bottom[3], color = 'purple', alpha=0.1)
    plt.fill_between(np.arange(0,10), top[4], bottom[4], color = 'yellow', alpha=0.1)
    plt.xticks(np.arange(0,10,1),np.arange(10,110,10))
    plt.xlabel(xlabel='Percentage of Training Data')
    plt.ylabel(ylabel='Accuracy')

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.show()
        
   # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanGraph[0][-1]
    stats[0,1] = stdGraph[0][-1]
    stats[1,0] = meanGraph[1][-1]
    stats[1,1] = stdGraph[1][-1]
    stats[2,0] = meanGraph[2][-1]
    stats[2,1] = stdGraph[2][-1]
    return stats

# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.