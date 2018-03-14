#coding=utf-8
import numpy as np

#加载数据
def loadDataSet(filename):
    dataMat = []
    f = open(filename,'r')
    for line in f.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#计算两个数据之间的距离
def disEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

#随机生成一个k行n列的矩阵，为k个质心点
def randCent(dataMat,k):
    dataMat = np.mat(dataMat)
    n = np.shape(dataMat)[1]
    centroids = np.mat(np.zeros((k,n)))
    # print dataMat
    for j in range(n):
        minJ = np.min(dataMat[:,j])
        maxJ = np.max(dataMat[:,j])
        rangeJ = float(maxJ-minJ)
        centroids[:,j] = minJ +rangeJ * np.random.rand(k,1) #np.random.rand(m,n) 生成m行n列的随机矩阵，所有元素是(0,1)之间的浮点数
    # print '-'*50
    return centroids

#普通k均值聚类算法
def kMeans(dataMat,k,disMeans=disEclud,createCent=randCent):
    dataMat = np.mat(dataMat)
    m = np.shape(dataMat)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataMat,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                disJI = disMeans(centroids[j,:],dataMat[i,:])
                if disJI < minDist:
                    minDist = disJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print centroids
        for cent in range(k):
            ptsInClust = dataMat[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust,0)
    return centroids,clusterAssment

#二分K-均值聚类算法
def biKmeans(dataMat,k,disMeas=disEclud):
    dataMat = np.mat(dataMat)
    m = np.shape(dataMat)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataMat,0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = disEclud(np.mat(centroid0),dataMat[j,:])**2
    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurCluster = dataMat[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            # print i
            # print ptsInCurCluster
            # print '-'*50
            centroidMat,splitClusterAss = kMeans(ptsInCurCluster,2,disMeas)
            # print splitClusterAss
            # print '-'*100
            sseSplit = np.sum(splitClusterAss[:,1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            if (sseNotSplit + sseSplit) < lowestSSE:
                lowestSSE = sseSplit + sseNotSplit
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClusterAss = splitClusterAss.copy()
        print bestCentToSplit
        bestClusterAss[np.nonzero(bestClusterAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClusterAss[np.nonzero(bestClusterAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print bestClusterAss
        print '-'*100
            # print 'the best cent to split',bestCentToSplit
            # print 'the len of bestClusterAss is:',bestClusterAss
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        # print bestClusterAss
        # print '-'*100
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClusterAss
        # print clusterAssment
        # print '-'*50
    return np.mat(centList),clusterAssment

if __name__ == '__main__':
    dataMat = loadDataSet('/home/szw/machinelearninginaction/Ch10/testSet.txt')
    centroids,clusterAssment = biKmeans(dataMat,3)
    print centroids
    # print centroids,clusterAssment