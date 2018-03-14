#coding=utf8

import pickle

import MyTree
import treePlotter

def classify(modelTree,nodes,testVec):
    firstNode = modelTree.keys()[0]
    nodeIndex = nodes.index(firstNode)
    secondTree = modelTree[firstNode]
    for key in secondTree.keys():
        if testVec[nodeIndex] == key:
            if type(secondTree[key]).__name__ == 'dict':
                classLabel = classify(secondTree[key],nodes,testVec)
            else:
                classLabel = secondTree[key]
    return classLabel

if __name__ == '__main__':
    # fr = open('/home/szw/machinelearninginaction/Ch03/lenses.txt')
    # dataSet = [line.strip().split('\t') for line in fr.readlines()]
    nodes = ['age','prescript','astigmatic','tearRate']
    # myTree = MyTree.createTree(dataSet,nodes)
    # print myTree
    # MyTree.storeTree(myTree,'firstTree')
    # treePlotter.createPlot(myTree)
    tree = MyTree.loadtree('firstTree')
    testVec = ['young','myope','no','reduced']
    print classify(tree,nodes,testVec)