import math
def q1():
    size = 14
    outlook = [0,0,0,0,0,1,1,1,1,2,2,2,2,2]
    temp = [75,80,85,72,69,72,83,64,81,71,65,75,68,70]
    hum =  [70,90,85,95,70,90,78,65,75,80,70,80,80,96]
    windy = [1,1,0,0,0,1,0,1,0,1,1,0,0,0]
    play = [1,0,0,0,1,1,1,1,1,0,0,1,1,1]

    #attributeSize = 4

    data = []
    for i in range(size):
        data.append((outlook[i], temp[i], hum[i], windy[i], play[i]))
    
    attributeDataMap = ["outlook", "temperature", "humidity", "windy"]
    attributeGroupFunction = [equals, lessEqual(75), lessEqual(75), equals]
    availAttr = set(attributeNames)
    outputData = []
    tempData = data
    tempOutputData = []
    # for attr in availAttr:
    attrIndex = attributeDataMap.index(attr)
    group(tempData, attrIndex, attributeGroupFunction[attrIndex])
    # availAttr 
    # getChildren(data)
    
def equals(val):
    return val

def lessEqual(bound):
    def func(val):
        if val <= bound:
            return 0
        else:
            return 1
    return func

def entropyCalc(tup):
    entropy = 0
    total = sum(tup)
    for v in tup:
        # print(v)
        # print(total)
        if v == 0:
            continue
        entropy += -(v/total) * math.log2(v/total)
    #print(tup, entropy)
    return entropy

def gain(children):
    total = sum([sum(tu) for tu in children])
    parent = [0,0]
    childSumWeightEntropy  = 0
    for child in children:
        parent[0] += child[0]
        parent[1] += child[1]
        print(entropyCalc(child))
        childSumWeightEntropy += (sum(child)/total) * entropyCalc(child)
    parentEntropy = entropyCalc(parent)
    gain = parentEntropy - childSumWeightEntropy
    return gain

if __name__ == "__main__":
    # filter = (0, 0)
    print(gain([(0,2), (3,0)]))
    # print(sorted([75,80,85,72,69,72,83,64,81,71,65,75,68,70]))
    # print(sorted([70,90,85,95,70,90,78,65,75,80,70,80,80,96]))
    #print(entropyCalc((3,3)))
    