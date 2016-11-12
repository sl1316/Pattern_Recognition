import numpy as np
from sklearn.decomposition import PCA
from numpy import matrix
import matplotlib.pyplot as plt

def drawVarianceDistribution(sample):
    plt.figure(figsize=(20, 20))
    d = [k + 1 for k in range(20)]
    variance_ratio = []
    for i in range(20):
        pca = PCA(n_components= (i + 1))
        tSample = np.copy(sample)
        pca.fit(sample)
        #print pca.explained_variance_ratio_
        print pca.explained_variance_ratio_.cumsum().max()
        #already sort in descending order
        variance_ratio.append(np.sum(pca.explained_variance_ratio_))

    plt.scatter(d, variance_ratio)
    plt.tight_layout()
    plt.show()
def findLowestComNum(sample):
    for i in range(20):
        pca = PCA(n_components= (i + 1))
        pca.fit(sample)
        if pca.explained_variance_ratio_.cumsum().max() > 0.95:
            return i + 1

def calCov(sample):
    return np.cov(sample)

def drawImage(mat,figureNum,title):
    plt.subplot(figureNum)
    plt.title(title)
    plt.pcolor(np.asarray(mat))
    plt.colorbar()

def inverseMatrix(mat):
    invMat = matrix(mat).I
    return invMat
def drawErrorPlot(reconData, origData, figureNum, title):
    def dist(a, b):
        return np.sum((a - b) ** 2)

    plt.subplot(figureNum)
    plt.title(title)
    train_error = []
    reconData = reconData.tolist()
    for i in range(len(reconData)):
        train_error.append(dist(reconData[i],origData[i]))
    x = [i + 1 for i in range(len(reconData))]
    plt.plot(x, train_error, '-')

sample = np.load("sample.npy")
trainingSample = sample[:750]
testingSample = sample[750:]

#get lowest ideal num of components
num_components = findLowestComNum(trainingSample)
pca = PCA(n_components=num_components)
pca.fit(trainingSample)

#draw main direction
print np.asarray(pca.components_).shape
drawImage(pca.components_,111,"Principal Directions")
plt.show()
pTrainSample = pca.transform(trainingSample)
covMat = calCov(pTrainSample.T)
precMat = inverseMatrix(covMat)
drawImage(covMat,221,"Covirance Matrix")
drawImage(precMat,222, "Precision Matrix")
reconTrainData = pca.inverse_transform(pTrainSample)
trainErrorMat = trainingSample - reconTrainData
drawErrorPlot(reconTrainData,trainingSample,223,"Train Reconstruction Error")

#test reconstruction
pTestSample = pca.transform(testingSample)
reconTestData = pca.inverse_transform(pTestSample)
drawErrorPlot(reconTestData, testingSample,224,"Test Reconstruction Error")

plt.tight_layout()
plt.show()