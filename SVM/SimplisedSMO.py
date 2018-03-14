#coding=utf8
import random
import numpy as np

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename,'r')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([np.float(lineArr[0]),np.float(lineArr[1])])
        labelMat.append(np.float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j = i
    while j==i:
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj

def smoSimple(dataMat,labelMat,C,toler,maxIter):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    b = 0
    m,n = np.shape(dataMat)
    alphas = np.mat(np.zeros((m,1)))
    iter  = 0
    while iter<maxIter:
        alphaParisChanged = 0
        for i in range(m):
            # Fxi = np.float(np.multiply(alphas,labelMat).T * (dataMat*dataMat[i,:].T)) + b
            Fxi = np.float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = Fxi - np.float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                Fxj = np.float(np.multiply(alphas,labelMat).T * (dataMat*dataMat[j,:].T)) + b
                Ej = Fxj - np.float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0,alphaJold-alphaIold)
                    H = min(C,C+alphaJold-alphaIold)
                else:
                    L = max(0,alphaIold+alphaJold-C)
                    H = min(C,alphaJold+alphaIold)
                if L == H:
                    print 'L equals H'
                    continue
                # eta = 2*dataMat[i]*dataMat[j].transpose()-dataMat[i]*dataMat[i].transpose()-dataMat[j]*dataMat[j].transpose()
                eta = 2.0 * dataMat[i,:]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
                if eta == 0:
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphaJold - alphas[j])) < 0.00001:
                    print 'alphaJ do not move enough'
                    continue
                alphas[i] += labelMat[i]*labelMat[j]*(alphaJold - alphas[j])
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[i,:]*dataMat[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T
                if (alphas[i]>0) and (alphas[i]<C):
                    b = b1
                elif (alphas[j]>0) and (alphas[j]<C):
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaParisChanged += 1
                print 'iter:%d i:%d paris changed:%d'%(iter,i,alphaParisChanged)
        if alphaParisChanged == 0:
            iter += 1
        else:
            iter = 0
        print 'iteration number %d' %iter
    return alphas,b

#Platt SMO完整算法
#定义一个保存数据的数据结构
class optStruct:
    def __init__(self,dataMat,labelMat,C,toler):
        self.dataMat = dataMat
        self.labelMat = labelMat
        self.C = C
        self.toler = toler
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))

def calcEk(oS,k):
    fxk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.dataMat*oS.dataMat[k,:].T)) + oS.b
    Ek = fxk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcahceList = np.nonzero(oS.eCache[:,0].A)[0]
    if len(validEcahceList) > 1:
        for k in validEcahceList:
            if k == i:continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ek-Ei)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
    else:
        maxK = selectJrand(i,oS.m)
        Ej = calcEk(oS,maxK)
    return maxK,Ej

def updata(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

#对于数据结构oS和某个i，先验证第i个alpha是否符合外层循环，如果符合，选择另一个alphaJ，并更新oS中alphaI、
# alphaJ、eCacheI、eCacheJ、b，如果发生列更新，返回1,否则返回0
def innerL(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.toler) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaIold + alphaJold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        if L == H:
            print 'L == H'
            return 0
        eta = 2.0 * oS.dataMat[i,:] * oS.dataMat[j,:].T - oS.dataMat[i,:] * oS.dataMat[i,:].T - oS.dataMat[j,:] * oS.dataMat[j,:].T
        if eta == 0:
            print 'eta =0'
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updata(oS,j)
        if abs(oS.alphas[j]-alphaJold)<0.00001:
            print 'j donnot move enough'
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updata(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.dataMat[i,:] * oS.dataMat[i, :].T - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.dataMat[i,:] * oS.dataMat[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.dataMat[i, :] * oS.dataMat[j, :].T - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.dataMat[j, :] * oS.dataMat[j,:].T
        if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif (oS.alphas[j] > 0) and (oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0

#完整的plattSMO算法
def smoP(dataMat,labelMat,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(np.mat(dataMat),np.mat(labelMat).transpose(),C,toler)
    iter = 0
    entireSet = True
    alphasParisChanged = 0
    while ((iter < maxIter) and (alphasParisChanged > 0)) or (entireSet):
        alphasParisChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphasParisChanged += innerL(i,oS)
                print 'fullSet,iter:%d,i:%i,parisChanged:%d' %(iter,i,alphasParisChanged)
            iter +=1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A>0) * (oS.alphas.A<oS.C))[0]
            for i in nonBoundIs:
                alphasParisChanged += innerL(i,oS)
                print 'non-bound,iter:%d,i:%d,parisChanged:%d' %(iter,i,alphasParisChanged)
            iter +=1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphasParisChanged == 0): entireSet = True
        print 'iterNumber:%d' %iter
    return oS.alphas,oS.b

def calcW(alphas,dataMat,labelMat):
    x = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    m,n = np.shape(x)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],x[i,:].T)
    return w

if __name__ == '__main__':
    dataMat,labelMat = loadDataSet('/home/szw/machinelearninginaction/Ch06/testSet.txt')
    alphas,b = smoP(dataMat,labelMat,0.6,0.001,40)
    print 'alphas b *******************************************************************'
    print alphas
    print b
    print '*'*30
    w = calcW(alphas,dataMat,labelMat)
    print 'w'
    print w