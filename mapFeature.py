import numpy as np
def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    # print(x1)
    # print(x2)
    n = len(x1)
    out = np.empty((n,1))
    for i in range(0, 7):
        # print("layer", i)
        for f1 in range(i+1):
            nextCol = np.multiply(np.power(x1, f1), np.power(x2, i-f1))
            # print(nextCol)
            out = np.c_[out, nextCol]
            # print(out)
    return out[:,1:]



