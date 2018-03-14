#coding=utf8
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numberClass import *



def getData():
    #把训练图像转化成classify可使用的格式
    trainingFileList = listdir('/home/szw/machinelearninginaction/Ch02/digits/trainingDigits')
    trainingFileCount = len(trainingFileList)
    traingData = np.zeros((trainingFileCount,1024))
    HWlabels = []
    for i in range(trainingFileCount):
        fileName = trainingFileList[i]
        fileVector = img2vector('/home/szw/machinelearninginaction/Ch02/digits/trainingDigits/{}'.format(fileName))
        fileInt = fileName.split('_')[0]
        traingData[i] = fileVector
        HWlabels.append(fileInt)
    testFileList = listdir('/home/szw/machinelearninginaction/Ch02/digits/testDigits')
    testFileCount = len(testFileList)
    testData = np.zeros((testFileCount,1024))
    testLabels = []
    for i in range(testFileCount):
        fileName = testFileList[i]
        fileVector = img2vector('/home/szw/machinelearninginaction/Ch02/digits/testDigits/{}'.format(fileName))
        fileInt = fileName.split('_')[0]
        testData[i] = fileVector
        testLabels.append(fileInt)
    return traingData,HWlabels,testData,testLabels

if __name__ == '__main__':
    time1 = time.time()
    traingData,traingLabels,testData,testLabels = getData()
    dic = {}
    for i in range(2,20):
        knn = KNeighborsClassifier(algorithm='kd_tree',n_neighbors=i)
        knn.fit(traingData,traingLabels)
        predictLabels = knn.predict(testData)
        errCount = 0
        errRate = 0
        labelDif = np.array(testLabels,dtype='S32').astype('float64')-predictLabels.astype('float64')
        for j in labelDif:
            if j != 0:
                errCount +=1
        print i,errCount
        errRate = float(errCount)/len(labelDif)
        dic[str(i)] = errRate
    sortedDic=sorted(dic.iteritems(),key=operator.itemgetter(1))#将得到的字典按count进行排序
    print '错误率最小的K值为'
    print sortedDic[0]
    time2 = time.time()
    print '测试一共用时'+str(time2-time1)

    # 错误率最小的K值为
    # ('3', 0.011627906976744186)
    # 测试一共用时57.5109801292



