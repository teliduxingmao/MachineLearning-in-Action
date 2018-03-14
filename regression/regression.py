#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    f = open(filename,'r')
    dataMat = []
    labelMat = []
    for line in f.readlines():
        datalist = line.strip().split('\t')
        dataMat.append([float(datalist[0]),float(datalist[1])])
        labelMat.append(float(datalist[-1]))
    f.close()
    return dataMat,labelMat

#常规线性回归
#ws = (X.T * X).I * X.T * Y
def standRrgres(dataMat,labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    xTx = dataMat.T*dataMat
    if np.linalg.det(xTx) == 0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I *(dataMat.T*labelMat)
    return ws

#局部加权线性回归
#ws = (X.T *W* X).I * X.T * W * Y
#其中W(i,i) = exp((xi-x)/(-2k**2))
def lwlr(testPoint,dataMat,labelMat,k = 1.0):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    m = np.shape(dataMat)[0]
    weights = np.mat(np.eye(m)) #构造对角矩阵
    for j in range(m):
        diffMat = dataMat[j,:] - testPoint
        weights[j,j] = np.exp(diffMat * diffMat.T/(-2*k**2))
    xTx = dataMat.T * (weights * dataMat)
    if np.linalg.det(xTx) == 0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I * (dataMat.T *(weights*labelMat))
    return testPoint.T * ws

#岭回归
def ridgeRegres(dataMat,labelMat,lam=0.2):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    xTx = dataMat.T * dataMat
    denom = xTx +np.eye(np.shape(dataMat)[1]) * lam
    if np.linalg.det(denom) == 0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = denom.I * (dataMat.T * labelMat)

#标准线性回归测试
# if __name__ == '__main__':
#     dataMat,labelMat = loadDataSet('/home/szw/machinelearninginaction/Ch08/ex0.txt')
#     ws = standRrgres(dataMat,labelMat)
#     dataMat = np.mat(dataMat)
#     labelMat = np.mat(labelMat)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(dataMat[:,1].flatten().A[0],labelMat.T[:,0].flatten().A[0])
#     dataMat.sort(0)
#     labelHat = dataMat * ws
#     ax.plot(dataMat[:,1],labelHat)
#     plt.show()

#局部加权线性回归测试
if __name__ == '__main__':
    dataMat,labelMat = loadDataSet('/home/szw/machinelearninginaction/Ch08/ex0.txt')
    m = np.shape(dataMat)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(np.array(dataMat[i]),dataMat,labelMat)
    print yHat

