#coding=utf8
import feedparser,operator
from bayes import *


def calMostFreq(vocabList,fullWords):
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullWords.count(word)
    sortedFreqDict = sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse = True)
    return sortedFreqDict[:30]

def localWords(feed1,feed0,testNum = 10):
    print 'start'+'-'*10
    sampleList = []
    classList = []
    fullWords = []
    #将所有文本数据加载到sampleList中，其类型放在classList中
    minlen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minlen):
        wordList = textParse(feed1['entries'][i]['summary'])
        fullWords.extend(wordList)
        sampleList.append(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        fullWords.extend(wordList)
        sampleList.append(wordList)
        classList.append(0)
    traingIndexes = range(0,2*minlen)
    print 'sample'
    print len(sampleList)
    testIndexes = []
    #将10条数据加到测试集中，剩下的作为训练集
    vocabList = createVocabList(sampleList)
    print vocabList
    #取出频数前30的单词
    top30Freq = calMostFreq(vocabList,fullWords)
    for word in top30Freq:
        if word in vocabList:
            vocabList.remove(word)
    for i in range(testNum):
        randIndex = random.randint(0,len(traingIndexes)-1)
        testIndexes.append(traingIndexes[randIndex])
        del traingIndexes[randIndex]
    traingMat = []
    traingClass = []
    for index in traingIndexes:
        traingMat.append(setOfWords2Vec(vocabList,sampleList[index]))
        traingClass.append(classList[index])
    p0Vec,p1Vec,pClass1 = trainNB0(traingMat,traingClass)
    errorCount = 0
    for index in testIndexes:
        wordVect = setOfWords2Vec(vocabList,sampleList[index])
        predictedClass = classifyNB(wordVect,p0Vec,p1Vec,pClass1)
        if predictedClass != classList[index]:
            errorCount +=1
    errorRate = errorCount/float(testNum)
    print errorCount
    return p0Vec,p1Vec,pClass1,errorRate

if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    testNum = 10
    p0Vec, p1Vec, pClass1, errorRate = localWords(ny,sf,testNum)
    print errorRate
