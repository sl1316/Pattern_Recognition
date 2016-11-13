## PR Midterm 2016 fall
## Xinyu Li (152000006)

import numpy as np
import math
from numpy import matrix
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from ppca import FactorAnalysis

#-----------------------------------------------------------------------------------------------
# calculate the probability
def calProbInMatrix():
    X = np.random.rand(20, 2)
    ProbMat = []
    for i in range(X.shape[0]):
        ProbV = []
        for j in range(X.shape[0]):
            tmp=(1 / math.sqrt(2 * math.pi)) * math.exp((-4) * np.sum((X[i] - X[j]) ** 2))
            ProbV.append(tmp)
        ProbMat.append(ProbV)
    return ProbMat

# calculate the adjescent matrix
def calAdjMatrix(ProbMat):
    adMat = (ProbMat > np.random.rand(len(ProbMat), len(ProbMat[1]))).astype(int)
    num_rows = len(adMat)
    for i in range(num_rows):
        for j in range(0, i):
            adMat[j][i] = adMat[i][j]
    return adMat

# calculate inverse covariance matrix
def calInverseCovMatrix(AdjMat):
    invCovMat = []
    num_rows = len(AdjMat)
    num_cols = len(AdjMat[0])
    for i in range(num_rows):
        covList = []
        for j in range(num_cols):
            if i == j:
                cov = 1
            elif AdjMat[i][j] == 1:
                cov = 0.245
            else:
                cov = 0
            covList.append(cov)
        invCovMat.append(covList)
    return invCovMat

# plot the final results
def plotFinalResult5(AdjMat, invCovMat,CovMat):
    # graph
    plt.subplot(131, aspect='equal')
    rows, cols = np.where(AdjMat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr)
    plt.title('Adjacency Matrix')
    # cov
    plt.subplot(132, aspect='equal')
    im1=plt.imshow(np.asarray(CovMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Precision Matrix')
    # Inverse
    plt.subplot(133, aspect='equal')
    im2 =plt.imshow(np.asarray(invCovMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('Covariance Matrix')
    plt.show()

# main for problem 2.5
PrMat = calProbInMatrix()
AdjMat = calAdjMatrix(PrMat)
invCovMat = calInverseCovMatrix(AdjMat)
CovMat = matrix(invCovMat).I
plotFinalResult5(AdjMat, invCovMat,CovMat)

#-----------------------------------------------------------------------------------------------
# plot the final results
def plotFinalResult6(CovMat, invCovMat, covarianceMat,precisionMat):
    # Model Covariance Matrix
    plt.subplot(221, aspect='equal')
    im1=plt.imshow(np.asarray(CovMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Model Covariance Matrix')
    # Sample Covariance Matrix
    plt.subplot(222, aspect='equal')
    im1=plt.imshow(np.asarray(invCovMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Model Procesion Matrix')
    # Sample Covariance Matrix
    plt.subplot(223, aspect='equal')
    im1=plt.imshow(np.asarray(covarianceMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Sample Covariance Matrix')
    # Sample Precision Matrix
    plt.subplot(224, aspect='equal')
    im2 =plt.imshow(np.asarray(precisionMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('Sample Precision Matrix')
    plt.show()

# main for problem 2.6
mean = [1.5 for i in range(20)]
sample = np.random.multivariate_normal(mean, CovMat, 1000)
covarianceMat = np.cov(sample.T)
precisionMat = matrix(covarianceMat).I
plotFinalResult6(CovMat, invCovMat, covarianceMat,precisionMat)

#-----------------------------------------------------------------------------------------------
# plot the final results
def plotFinalResult7(components, components_, covarianceMatTrain, pcaPrecisionMat, pcaTrainError, pcaTestError):
    # component
    plt.subplot(321, aspect='equal')
    im1=plt.imshow(np.asarray(components), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Principal Directions')
    # component
    plt.subplot(322)
    x = [i + 1 for i in range(len(components_))]
    plt.plot(x, components_, '-')
    plt.title('Principal Directions')
    # Sample Covariance Matrix
    plt.subplot(323, aspect='equal')
    im1=plt.imshow(np.asarray(covarianceMatTrain), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Covirance Matrix')
    # Sample Precision Matrix
    plt.subplot(324, aspect='equal')
    im2 =plt.imshow(np.asarray(pcaPrecisionMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('Precision Matrix')
    plt.subplot(325)
    x = [i + 1 for i in range(750)]
    plt.plot(x, pcaTrainError, '-')
    plt.title('Train Reconstruction Error')
    # Sample Precision Matrix
    plt.subplot(326)
    x = [i + 1 for i in range(250)]
    plt.plot(x, pcaTestError, '-')
    plt.title('Test Reconstruction Error')
    plt.show()

# main for problem 2.7
trainingSample = sample[:750]
testingSample = sample[750:]

# determine the num fo components
for i in range(20):
    pca = PCA(n_components=(i + 1))
    pca.fit(trainingSample)
    if pca.explained_variance_ratio_.cumsum().max() > 0.95:
        num_components=i+1
        break

pca = PCA(n_components=num_components)
pca.fit(trainingSample)
components=pca.components_
components_=pca.explained_variance_ratio_

# training
pcaTrainSample = pca.transform(trainingSample)
covarianceMatTrain = np.cov(pcaTrainSample.T)
pcaPrecisionMat = matrix(covarianceMatTrain).I
pcaTrainSampleInv = pca.inverse_transform(pcaTrainSample)
pcaTrainError= np.sqrt(np.sum((trainingSample-pcaTrainSampleInv)**2,axis=1))

# testing
pcaTestSample = pca.transform(testingSample)
pcaTestSampleInv = pca.inverse_transform(pcaTestSample)
pcaTestError= np.sqrt(np.sum((testingSample-pcaTestSampleInv)**2,axis=1))
plotFinalResult7(components,components_, covarianceMatTrain, pcaPrecisionMat, pcaTrainError, pcaTestError)

#-----------------------------------------------------------------------------------------------
def plotFinalResult8(componentsPPCA, covariancePPCAMatTrain, ppcaPrecisionMat, ppcaTrainError, ppcaTestError, pcaTrainError, pcaTestError):
    # component
    plt.subplot(321, aspect='equal')
    im1=plt.imshow(np.asarray(componentsPPCA), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Principal Directions')
    # Sample Covariance Matrix
    plt.subplot(323, aspect='equal')
    im1=plt.imshow(np.asarray(covariancePPCAMatTrain), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Covirance Matrix')
    # Sample Precision Matrix
    plt.subplot(324, aspect='equal')
    im2 =plt.imshow(np.asarray(ppcaPrecisionMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('Precision Matrix')
    plt.subplot(325)
    x = [i + 1 for i in range(750)]
    plt.plot(x, ppcaTrainError, '-')
    plt.title('Train Reconstruction Error')
    # Sample Precision Matrix
    plt.subplot(326)
    x = [i + 1 for i in range(250)]
    plt.plot(x, ppcaTestError, '-')
    plt.title('Test Reconstruction Error')
    plt.show()

    plt.subplot(121)
    x = [i + 1 for i in range(750)]
    plt.plot(x, ppcaTrainError, 'r-', label='training error for PPCA')
    plt.plot(x, pcaTrainError, 'b-', label='training error for PCA')
    plt.legend()
    plt.title('Comparison of Train Reconstruction Error')
    # Sample Precision Matrix
    plt.subplot(122)
    x = [i + 1 for i in range(250)]
    plt.plot(x, ppcaTestError, 'r-',label='testing error for PPCA')
    plt.plot(x, pcaTestError, 'b-',label='testing error for PCA')
    plt.legend()
    plt.title('Comparison Test Reconstruction Error')
    plt.show()

# main for problem 2.8

num_components = 0
ppcaModelTmp = FactorAnalysis(n_components=20)
ppcaModelTmp.fit(trainingSample)
s,v,d = np.linalg.svd(ppcaModelTmp.components_)
totalVariance = 0

for i in range(len(ppcaModelTmp.get_covariance())):
    totalVariance += v[i]
totalVarRatio = 0
for i in range(len(ppcaModelTmp.get_covariance())):
    if totalVarRatio < 0.95:
        num_components+=1
        totalVarRatio+=(v[i]/totalVariance)

ppcaModel = FactorAnalysis(n_components=num_components+1)
ppcaModel.fit(trainingSample)

# train
ppcaModel.fit(trainingSample)
componentsPPCA=ppcaModel.components_
ppcaTrainSample = ppcaModel.transform(trainingSample)
covariancePPCAMatTrain = np.cov(ppcaTrainSample.T)
ppcaPrecisionMat = matrix(covariancePPCAMatTrain).I
ppcaTrainSampleInv = ppcaModel.inver_transform(ppcaTrainSample)
ppcaTrainError= np.sqrt(np.sum((trainingSample-ppcaTrainSampleInv.tolist())**2,axis=1))

# test
ppcaTestSample = ppcaModel.transform(testingSample)
ppcaTestSampleInv = ppcaModel.inver_transform(ppcaTestSample)
ppcaTestError= np.sqrt(np.sum((testingSample-ppcaTestSampleInv.tolist())**2,axis=1))

plotFinalResult8(componentsPPCA, covariancePPCAMatTrain, ppcaPrecisionMat, ppcaTrainError, ppcaTestError,pcaTrainError, pcaTestError)