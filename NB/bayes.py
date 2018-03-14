#coding=utf8
#词袋模型不符合统计学习方法中P(X=x|Y=ci)的计算公式，故本次没有使用，感觉《机器学习实战》这本书问题很大。
import numpy as np
import re,random


def loadDataSet():
    postingList = [
        ['my','dog','has','problems','help','please'],
        ['maybe', 'not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    classVec = [0,1,0,1,0,1]    #0代表正常言论，1代表侮辱性言论
    return postingList,classVec

def createVocabList(dataList):
    vocabList = set([])
    for data in dataList:
        vocabList = vocabList | set(data)
    return list(vocabList)

def setOfWords2Vec(vocabList,inputSet):
    wordVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            wordVec[vocabList.index(word)] = 1
        else:
            print '%s is not in my vocabylary'%word
    return wordVec

def trainNB0(trainMatrix,classVec):
    numOfData = len(trainMatrix)
    lenOfVec = len(trainMatrix[0])
    pClass1 = (sum(classVec)+1)/float(numOfData+2)  #此侮辱性文本的概率，防止概率为0,采用了贝叶斯估计，采用拉斯平滑，k=2,lambda=1
    # p0Vec = np.zeros(lenOfVec)
    # p1Vec = np.zeros(lenOfVec)    #构造一个与特征向量维数相同的向量来统计每个特征发生的概率
    p0Vec = np.ones(lenOfVec)
    p1Vec = np.ones(lenOfVec)    #采用贝叶斯估计，入=1
    # p0Demon = 0
    # p1Demon = 0         #分类1和0的分母
    p0Demon = 2
    p1Demon = 2    #为防止出现概率为0的情况，此处采用贝叶斯估计，此时令lambda=1,每个特征只有2个可能，0或1
    for i in range(numOfData):
        if classVec[i] == 0:
            p0Vec += trainMatrix[i]
            p0Demon += 1
        else:
            p1Vec += trainMatrix[i]
            p1Demon += 1        #此处与机器学习实战中源码不同，原书错误，此为参考李航《统计学习方法》
    p0Vec = np.log(p0Vec/float(p0Demon))
    p1Vec = np.log(p1Vec/float(p1Demon))   #将所有的概率进行log运算，防止概率相乘后太小导致计算机运算误差太大,
                                            # 相应的，后续概率概率计算变为相加
    return p0Vec,p1Vec,pClass1   #p0Vec,p1Vec为特征向量的条件概率分布向量

def classifyNB(testVec,p0Vec,p1Vec,pClass1):
    p1 = sum(testVec*p1Vec)+pClass1       #此处的testVec和p0Vec、p1Vec均为numpy.array,相乘时对应的元素相乘
    p0 = sum(testVec*p0Vec)+1-pClass1
    if p1 > p0:
        return 1
    else:
        return 0

def predict(testData):
    dataList, classVec = loadDataSet()
    vocabList = createVocabList(dataList)
    dataMatrix = []
    for i in range(len(dataList)):
        dataMatrix.append(setOfWords2Vec(vocabList, dataList[i]))
    p0Vec, p1Vec, pClass1 = trainNB0(dataMatrix, classVec)
    testVec = setOfWords2Vec(vocabList,testData)
    classOfTestData = classifyNB(testVec,p0Vec,p1Vec,pClass1)
    return classOfTestData

def textParse(bigString):
    wordList = re.split(r'\W*',bigString)
    return [word.lower() for word in wordList if len(word)>2]

def spamTest():
    dataList = []    #用于存放从文本中解析出来的wordList
    classList = []   #用于存放从文本对应的类型
    #将所有文本数据加载到dataList中，其类型放在classList中
    for i in range(1,26):
        with open('/home/szw/machinelearninginaction/Ch04/email/spam/{}.txt'.format(i)) as f:
            bigString = f.read()
            dataList.append(textParse(bigString))
            classList.append(1)
        with open('/home/szw/machinelearninginaction/Ch04/email/ham/{}.txt'.format(i)) as f:
            bigString = f.read()
            dataList.append(textParse(bigString))
            classList.append(0)
    traingIndexes = range(0,50)  #一共50条数据
    testIndexes = []
    #将10条数据加到测试集中，剩下的作为训练集
    for i in range(10):
        randIndex = random.randint(0,len(traingIndexes)-1)
        testIndexes.append(traingIndexes[randIndex])
        del traingIndexes[randIndex]
    vocabList = createVocabList(dataList)
    traingMat = []
    traingClass = []
    for index in traingIndexes:
        traingMat.append(setOfWords2Vec(vocabList,dataList[index]))
        traingClass.append(classList[index])
    p0Vec,p1Vec,pClass1 = trainNB0(traingMat,traingClass)
    errorCount = 0
    for index in testIndexes:
        wordVect = setOfWords2Vec(vocabList,dataList[index])
        predictedClass = classifyNB(wordVect,p0Vec,p1Vec,pClass1)

        if predictedClass != classList[index]:
            errorCount +=1
    return errorCount/float(10)
if __name__ == '__main__':
    errRate = spamTest()
    print errRate