import numpy as np
from sklearn.decomposition import FactorAnalysis

from numpy import matrix
import matplotlib.pyplot as plt
def calCov(sample):
    return np.cov(sample)

def drawImage(mat,figureNum):
    plt.subplot(figureNum)
    plt.pcolor(np.asarray(mat))
    plt.colorbar()

def inverseMatrix(mat):
    invMat = matrix(mat).I
    return invMat

sample = np.load("sample.npy")
trainingSample = sample[:750]
testingSample = sample[750:]
fa = FactorAnalysis(n_components=20)
fa.fit(trainingSample)
ppTrainSample = fa.transform(trainingSample)
print ppTrainSample
covMat = calCov(ppTrainSample.T)
precMat = inverseMatrix(covMat)
print np.asarray(fa.score_samples(trainingSample)).shape
drawImage(covMat,221)
drawImage(precMat,222)
plt.show()