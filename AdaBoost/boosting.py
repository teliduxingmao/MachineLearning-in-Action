#coding=utf8
import numpy as np

def loadSimpData():
    dataMat = np.asmatrix([
        [1.,2.1],
        [2.,1.1],
        [1.3,1.],
        [1.,1.],
        [2.,1.]
    ])
    labelMat = [1,1,-1,-1,1]
    return dataMat,labelMat

def stumpClassify(dataMat,column,decisionPoint,inequal):
    predictedLabelMat = np.ones((np.shape(dataMat)[0],1))
    if inequal == 'lt':
        predictedLabelMat[dataMat[:,column]<=decisionPoint] = -1
    else:
        predictedLabelMat[dataMat[:,column]>decisionPoint] = -1
    return predictedLabelMat

def buildStump(dataMat,labelMat,D):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    m,n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestPredictedLabelMat = np.mat(np.zeros((m,1)))
    minErrRate = np.inf
    for column in range(n):
        rangeMin = dataMat[:,column].min()
        rangeMax = dataMat[:,column].max()
        stepSize = ((rangeMax-rangeMin)/numSteps)
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                decisionPoint = rangeMin + float(j)*stepSize
                predictedLabelMat = stumpClassify(dataMat,column,decisionPoint,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedLabelMat == labelMat] = 0
                weightedErr = D.T*errArr
                # print 'column:%d,decisionPoint:%.2f,inqual:%s,weightErr:%.3f'%(column,decisionPoint,inequal,weightedErr)
                if weightedErr < minErrRate:
                    minErrRate = weightedErr
                    bestPredictedLabelMat = predictedLabelMat.copy()
                    bestStump['column'] = column
                    bestStump['decisionPoint'] = decisionPoint
                    bestStump['inequal'] = inequal
    return bestStump,minErrRate,bestPredictedLabelMat

def adaBoostTrainDS(dataMat,labelMat,numIt=40):
    weakClassArr = []
    m = np.shape(dataMat)[0]
    D = np.mat(np.ones((m,1))/m)   #init D to all equal
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,errRate,predictedLabelMat = buildStump(dataMat,labelMat,D)#build Stump
        #print "D:",D.T
        alpha = np.float(0.5*np.log((1.0-errRate)/max(errRate,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1*alpha*np.mat(labelMat).T,predictedLabelMat) #exponent for D calc, getting messy
        D =np. multiply(D,np.exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*predictedLabelMat
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(labelMat).T,np.ones((m,1)))
        finalErrorRate = aggErrors.sum()/m
        # print "total errorRate: ",finalErrorRate
        if finalErrorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['column'],\
                                 classifierArr[i]['decisionPoint'],\
                                 classifierArr[i]['inequal'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print aggClassEst
    return np.sign(aggClassEst)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        if float(curLine[-1]) == 1:
            labelMat.append(1)
        else:
            labelMat.append(-1)
    return dataMat,labelMat

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = np.sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    print sortedIndicies.tolist()
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep

if __name__ == '__main__':
    # trainingDataMat,trainingLabelMat = loadDataSet('/home/szw/machinelearninginaction/Ch05/horseColicTraining.txt')
    # classifierArr = adaBoostTrainDS(trainingDataMat,trainingLabelMat,100)
    # testDataMat,testLabelMat = loadDataSet('/home/szw/machinelearninginaction/Ch05/horseColicTest.txt')
    # prediction = adaClassify(testDataMat,classifierArr)
    # print '*'*100
    # errArr = np.mat(np.ones((67,1)))
    # errRate = errArr[prediction != np.mat(testLabelMat).T].sum()
    # print errRate
    trainingDataMat,trainingLabelMat = loadDataSet('/home/szw/machinelearninginaction/Ch05/horseColicTraining.txt')
    classifierArr,aggClassEst = adaBoostTrainDS(trainingDataMat,trainingLabelMat,100)
    plotROC(aggClassEst.T,trainingLabelMat)
