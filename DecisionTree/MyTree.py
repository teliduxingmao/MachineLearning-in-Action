#coding=utf8
'''ID3算法的实现，不包括剪枝'''
from math import log
import numpy as np
import operator,pydotplus,pickle

#所有函数的dataSet最后一列为分类label
def calcShannonEnt(dataSet):
    shannonEnt = 0.0
    dataCounts = len(dataSet)
    labelDict = {}
    for featLabel in dataSet:
        currentLabel = featLabel[-1]
        if currentLabel not in labelDict:
            labelDict[currentLabel] = 0
        labelDict[currentLabel] +=1
    for key in labelDict:
        prob = float(labelDict[key])/dataCounts
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def splitSet(dataSet,axis,value):
    retDataSet = []
    for example in dataSet:
        if example[axis] == value:
            retDataSet.append(np.append(example[:axis],example[axis+1:]))
    return np.array(retDataSet)

def getBestFeature(dataSet):
    numFeature = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        newEntropy = 0.0
        featureList = [example[i] for example in dataSet]
        featureSet = set(featureList)     #转化为set去重
        for value in featureSet:
            retDataSet = splitSet(dataSet,i,value)
            prob = len(retDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(retDataSet)
        infoGain = baseEntropy-newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] +=1
    classCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return classCount[0][0]


#此处的labels为每个特征对应节点的名称
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #如果只有一类每直接返回
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    #如果没有特征向量了，返回投票最多的类名
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    bestFeature = getBestFeature(dataSet)
    #获取节点名称
    labels = labels[:]      #拷贝labels，不改变labels，因为labels在分类的时候还要使用
    nodeName = labels[bestFeature]
    del labels[bestFeature]
    tree = {nodeName:{}}
    featureValues = [example[bestFeature] for example in dataSet]
    featureValues = set(featureValues)
    for value in featureValues:
        tree[nodeName][value] = createTree(splitSet(dataSet,bestFeature,value),labels)
    return tree


def storeTree(tree, filename):
    f = open(filename, 'w')
    pickle.dump(tree, f)
    f.close()

def loadtree(filename):
    f = open(filename, 'r')
    tree = pickle.load(f)
    f.close()
    return tree


if __name__ == '__main__':
    dataSet = np.array([[1,2,1,4  ],[2,2,2,3],[1,1,1,1],[4,1,3,0]])
    nodes = ['a','b','c']
    tree = loadtree('firstTree')
    print tree



