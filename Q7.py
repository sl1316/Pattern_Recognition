import numpy as np
from sklearn.decomposition import PCA
import random
from numpy import matrix
import matplotlib.pyplot as plt
def drawVarianceDistribution():
    plt.figure(figsize=(20, 20))
    d = [k + 1 for k in range(20)]
    variance_ratio = []
    for i in range(20):
        pca = PCA(n_components= (i + 1))
        tSample = np.copy(sample)
        pca.fit(sample)
        #print pca.explained_variance_ratio_
        #print pca.explained_variance_ratio_.cumsum().max()
        #already sort in descending order
        variance_ratio.append(np.sum(pca.explained_variance_ratio_))

    plt.scatter(d, variance_ratio)
    plt.tight_layout()
    plt.show()

def calCov(sample):
    return np.cov(sample)

def drawImage(mat,figureNum):
    plt.subplot(figureNum)
    #plt.imshow(np.asarray(mat), interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5, 10.5, 0.5, 10.5))
    plt.pcolor(np.asarray(mat))
    plt.colorbar()
def inverseMatrix(invCovMat):
    CovMat = matrix(invCovMat).I
    return CovMat
sample = np.load("sample.npy")
trainingSample = sample[:750]
testingSample = sample[750:]
print np.asarray(testingSample).shape
covMat = calCov(np.asarray(trainingSample).T)
print covMat.shape
invCovMat = inverseMatrix(covMat)
#drawImage(covMat,221)
#drawImage(invCovMat,222)
#drawVarianceDistribution()
pca = PCA(n_components=18)

pca.fit(trainingSample)
pTrainSample = pca.transform(trainingSample)
reconTrainData = pca.inverse_transform(pTrainSample)
trainErrorMat = trainingSample - reconTrainData
drawImage(trainErrorMat, 221)

pca.fit(testingSample)
pTestSample = pca.transform(testingSample)
reconTestData = pca.inverse_transform(pTestSample)
testErrorMat = testingSample - reconTestData
drawImage(testErrorMat,222)

plt.tight_layout()
plt.show()