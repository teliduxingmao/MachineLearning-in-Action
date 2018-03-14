#coding=utf8
from kNN import *
from os import listdir
import time
#把文本中的图像转化成向量
def img2vector(filename):
    returnVector = np.zeros((1,1024))
    f=open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVector[0,32*i+j] = lineStr[j]

    return returnVector

def handwrittenNumberClassTest(n):
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
    #转化完成
    #开始对预测集进行测试
    testFileList = listdir('/home/szw/machinelearninginaction/Ch02/digits/testDigits')
    errCount = 0
    total = len(testFileList)
    for fileName in testFileList:
        fileVector = img2vector('/home/szw/machinelearninginaction/Ch02/digits/testDigits/{}'.format(fileName))
        predictLabel = int(classify(fileVector,traingData,HWlabels,n))
        realLabel = int(fileName.split('_')[0])
        print '预测的数字是%d，实际的数字是%d'%(predictLabel,realLabel)
        if predictLabel != realLabel:
            errCount +=1
    print '一共的次数是%d，错误的次数是%d，错误率是%f'%(total,errCount,errCount/float(total))
    return errCount/float(total)

if __name__=='__main__':
    time1 = time.time()
    dic = {}
    for i in range(2,20):
        errRate = handwrittenNumberClassTest(i)
        dic[str(i)] = errRate
    print dic
    sortedDic=sorted(dic.iteritems(),key=operator.itemgetter(1))#将得到的字典按count进行排序
    print '错误率最小的K值为'
    print sortedDic[0]
    time2 = time.time()
    print '测试一共用时'+str(time2-time1)

    # 错误率最小的K值为
    # ('3', 0.012684989429175475)
    # 测试一共用时296.81005621