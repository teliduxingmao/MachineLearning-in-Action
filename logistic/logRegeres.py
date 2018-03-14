#coding=utf8
import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    with open('/home/szw/machinelearninginaction/Ch05/testSet.txt','r') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#梯度下降法
def gradAscent(dataMat,labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()   #将列表转化为numpy数组
    m,n = np.shape(dataMat)
    alpha = 0.001
    weights = np.ones((n,1))
    maxCycle = 1000
    for i in range(maxCycle):
        h = sigmoid(dataMat*weights)
        err = labelMat-h
        weights = weights + alpha*dataMat.transpose()*err
        print weights
    return weights

#随机梯度下降法
def randGradAscent(dataMat,labelMat):
    m,n = np.shape(dataMat)
    weights = np.ones(n)
    alpha = 0.005
    for i in range(m):
        h = sigmoid(np.sum(dataMat[i])*weights)
        err = labelMat[i] - h
        weights = weights + alpha * err * dataMat[i]
    return weights

#改进的随机梯度下降法，可应用在数据量较少的时候
def impRandGradAscent(dataMat,labelMat,cycle = 150):
    m,n = np.shape(dataMat)
    weights = np.ones(n)
    for i in range(cycle):
        indexList = range(m)
        for j in range(m):
            alpha = 1/(i+j+1)+0.01
            randIndex = random.randint(0,len(indexList)-1)
            h = sigmoid(np.sum(dataMat[randIndex]*weights))
            err = labelMat[randIndex] - h
            weights = weights + alpha*err*dataMat[randIndex]
            del indexList[randIndex]
    return weights

def classify(inX,weights):
    p1 = sigmoid(np.sum(inX*weights))
    if p1 >= 0.5:
        return 1
    else:
        return 0

def colickTest():
    frTrain = open('/home/szw/machinelearninginaction/Ch05/horseColicTraining.txt','r')
    frTest = open('/home/szw/machinelearninginaction/Ch05/horseColicTest.txt', 'r')
    trainingMat = []
    traingLabelsMat = []
    for line in frTrain.readlines():
        arr1 = []
        arr = line.strip().split()
        for i in range(21):
            arr1.append(float(arr[i]))
        trainingMat.append(arr1)
        traingLabelsMat.append(float(arr[21]))
    weights = impRandGradAscent(np.array(trainingMat),traingLabelsMat)
    numTest = 0
    numErr = 0
    for line in frTest.readlines():
        numTest += 1
        testVec = []
        vec = line.strip().split()
        for i in range(21):
            testVec.append(float(vec[i]))
        if int(classify(testVec,weights)) != int(vec[-1]):
            numErr +=1
    errRate = numErr/float(numTest)
    print 'error rate is %f'%errRate
    return errRate

def mutiTest(number):
    numErr = 0
    for i in range(number):
        numErr += colickTest()
    numErr = numErr/float(number)
    print 'after %d interations the average error rate is %f '%(number,numErr)


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    print 'last weight'
    print weights
    # plotBestFit(weights.getA())  #将weights矩阵转化为array
    # weights = randGradAscent(np.array(dataMat),labelMat)
    # weights = impRandGradAscent(np.array(dataMat), labelMat)
    # print 'weights%'
    # print weights
    # plotBestFit(weights)
    # mutiTest(10)