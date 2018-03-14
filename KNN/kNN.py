#coding=utf8
import numpy as np
import operator

#创建原始数据
def createSet():
    group=np.array([[1,0],[1.1,0],[0,1],[0.1,1]])
    labels=['A','A','B','B']
    return group,labels

#K-近邻算法分类（线性计算，未使用kd数）
def classify(inX,dataSet,labels,k):         #inX为输入向量，dataSet为数据集，labels为数据集对应的分类标签，k为选择的近邻数
    dataSetSize=dataSet.shape[0]            #获取数据集的行数
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet    #把inX转化为与dataSet相同行数的二维矩阵，再与dataSet相减，
    sqDiffMat=diffMat**2                    #平方
    sqDistances=sqDiffMat.sum(axis=1)       #按行求和
    distances=sqDistances**0.5              #求距离
    sortedDistances=distances.argsort()     #排序，返回的是由距离的索引组成的数组
    classCounts={}
    for i in range(k):
        voteIlabel=labels[sortedDistances[i]]   #获取最近的k个点的label
        classCounts[voteIlabel] = classCounts.get(voteIlabel,0)+1   #构造key=label：value=count的字典
    sortedClassCounts=sorted(classCounts.iteritems(),key=operator.itemgetter(1),reverse=True)#将得到的字典按count进行排序
    type=sortedClassCounts[0][0]    #获取count最大的lebel
    return type

#将文本中的数据转化成需要的二维数组
def file2matrix(filename):
    f=open(filename)
    arrayLines=f.readlines()
    f.close()
    numberOfLines=len(arrayLines)
    returnMat=np.zeros((numberOfLines,3))  #构建二维数组
    classLabelVectors=[]
    index=0
    for line in arrayLines:
        line=line.strip()
        listFromFile=line.split('\t')
        returnMat[index,:]=listFromFile[0:3]    #前三个是特征值
        classLabelVectors.append(int(listFromFile[-1])) #最后一个是类型，txt中数字为32位，转化为int，否则在做图用作颜色/大小参数时，会出现类型错误
        index+=1
    return returnMat,classLabelVectors

#归一化特征值，消除不同特征向量值大小的影响
def autoNormal(dataSet):
    minVals=dataSet.min(0)  #按列获取最小值
    maxVals=dataSet.max(0)
    ranges = maxVals-minVals
    m=dataSet.shape[0]  #获取dataSet的行数
    normalDataMat = dataSet - np.tile(minVals,(m,1)) #取dataSet与最小值的差值
    normalDataMat = normalDataMat/np.tile(ranges,(m,1))
    print ranges
    return normalDataMat,ranges,minVals

def datingClassTest():
    dataSet,classLabels = file2matrix('/home/szw/machinelearninginaction/Ch02/datingTestSet2.txt')
    hoRatio=0.20   #用最后的30%数据用来测试
    dataSet,ranges,minVals=autoNormal(dataSet)
    m=dataSet.shape[0]
    #取前80%作为训练集
    traingCount=int(m*(1-hoRatio))
    trainingData=dataSet[:traingCount+1]
    traingLabels=classLabels[:traingCount+1]
    testData=dataSet[traingCount+1:]
    testLabels=classLabels[traingCount+1:]
    errorCount = 0
    for i in range(len(testData)):
        data = testData[i]
        predictLabel = int(classify(data,trainingData,traingLabels,3)) #k=5
        realLabel = int(testLabels[i])
        print '分类返回的是%d,实际分类是%d'%(predictLabel,realLabel)
        if predictLabel != realLabel:
            errorCount +=1
    print '测试次数是%d,错误次数是%d，错误率是%f'%(len(testData),errorCount,float(errorCount)/len(testData))



if __name__=='__main__':
    # datingClassTest()
    dataSet, classLabels = file2matrix('/home/szw/machinelearninginaction/Ch02/datingTestSet2.txt')
    dataSet, ranges, minVals = autoNormal(dataSet)
    a=float(input('请输入里程'))
    b=float(input('请输入视频游戏所占时间比'))
    c=float(input('请输入每周冰淇淋公升数'))
    inX=np.array([a,b,c])
    type=classify(inX,dataSet,classLabels,2)
    print type
