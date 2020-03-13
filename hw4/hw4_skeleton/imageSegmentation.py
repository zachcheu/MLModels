from PIL import Image
import numpy as np
import math
import sys

def closestClusterIndex(clusters, feature):
    closestDiff = sys.maxsize
    closestClusterIndex = -1
    for i,c in enumerate(clusters):
        diff = (np.absolute(c - feature)).sum()
        if diff < closestDiff:
            closestDiff = diff
            closestClusterIndex = i
    return closestClusterIndex
    
def diff(clusters1, clusters2):
    diff = 0
    for i in range(len(clusters1)):
        # maybe just use x,y for difference
        diff += (np.absolute(clusters1[i] - clusters2[i])).sum()
    print(diff)
    return diff

if __name__ == "__main__":
    epsilon = 0.01
    assert(len(sys.argv) == 4)
    k = int(sys.argv[1])
    inFile = sys.argv[2]
    outFile = sys.argv[3]
    im = Image.open(inFile)
    imgV = np.array(im)
    imgV = imgV[:,:,:3]
    x,y,z = imgV.shape
    imgFeature = np.empty((0,5),int)
    for i in range(x):
        for j in range(y):
            temp = np.array([[i, j, imgV[i,j,0], imgV[i,j,1], imgV[i,j,2]]])
            imgFeature = np.append(imgFeature, temp, axis=0)

    n = imgFeature.shape[0]
    clusters = []
    for i in range(k):
        clusters.append(imgFeature[np.random.choice(n,1)])

    newClusters = None 
    iteration = 0
    while newClusters is None or diff(newClusters, clusters) > epsilon:
        if newClusters is not None:
            clusters = newClusters
        print("iteration: ", iteration)
        iteration+=1
        indexCounter = [0] * k
        indexSum = [[0,0,0,0,0] for i in range(k)]
        for i in range(n):
            index = closestClusterIndex(clusters, imgFeature[i])
            indexCounter[index] += 1
            indexSum[index][0] += imgFeature[i][0]
            indexSum[index][1] += imgFeature[i][1]
            indexSum[index][2] += imgFeature[i][2]
            indexSum[index][3] += imgFeature[i][3]
            indexSum[index][4] += imgFeature[i][4]

            
        newClusters = []
        for i,coord in enumerate(indexSum):
            if indexCounter[i] == 0:
                newClusters.append(imgFeature[np.random.choice(n,1)])
                continue
            x = math.floor(coord[0]/indexCounter[i])
            y = math.floor(coord[1]/indexCounter[i])
            r = math.floor(coord[2]/indexCounter[i])
            g = math.floor(coord[3]/indexCounter[i])
            b = math.floor(coord[4]/indexCounter[i])
            print(x,y,i)
            newClusters.append(np.array([x,y,r,g,b]).reshape((1,5)))
            
    clusters = [np.copy(i) for i in newClusters]

    npNewImage = np.copy(imgV)

    for i in range(x):
        for j in range(y):
            index = closestClusterIndex(clusters, np.array([x,y,imgV[x,y,0],imgV[x,y,1],imgV[x,y,2]]) )
            print(index)
            print(clusters[index])
            print(clusters[index].shape)
            npNewImage[x,y] = clusters[index][:,2:]
            
    print(npNewImage.shape)
    newImg = Image.fromarray(npNewImage)
    newImg.save(outFile)


    

    
