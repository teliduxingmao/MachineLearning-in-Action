#coding=utf8
from kNN import *
import matplotlib.pyplot as plt

datingDataMat,datingLabels=file2matrix('/home/szw/datingTestSet2.txt')
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(325) #图像在3行2列的网格中的第5个

#设置x，y轴数据，点的大小，颜色
ax.scatter(datingDataMat[:,1],datingDataMat[:,0],15*np.array(datingLabels),15*np.array(datingLabels))
plt.show()