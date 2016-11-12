## PR Midterm 2016 fall
## Xinyu Li (152000006)

import numpy as np
import math
from numpy import matrix
import matplotlib.pyplot as plt
import networkx as nx


# sampling fuction
def uniformSample():
    return np.random.rand(20, 2)

# calculate the probability
def calProbInMatrix(sample):
    def calProb(x, y):
        def dist(a, b):
            return np.sum((a - b) ** 2)
        return (1 / math.sqrt(2 * math.pi)) * math.exp((-4) * dist(x, y))

    X = sample
    M = X.shape[0]
    ProbMat = []
    for i in range(M):
        ProbV = []
        for j in range(M):
            ProbV.append(calProb(X[i], X[j]))
        ProbMat.append(ProbV)
    return ProbMat

#
def calAdMatrix(ProbMat):
    adMat = (ProbMat > np.random.rand(len(ProbMat), len(ProbMat[1]))).astype(int)
    num_rows = len(adMat)
    for i in range(num_rows):
        for j in range(0, i):
            adMat[j][i] = adMat[i][j]
    return adMat

#calculate inverse covariance matrix from adjacency matrix
def calInverseCovMatrix(AdMat):
    invCovMat = []
    num_rows = len(AdMat)
    num_cols = len(AdMat[0])
    for i in range(num_rows):
        covList = []
        for j in range(num_cols):
            if i == j:
                cov = 1
            elif AdMat[i][j] == 1:
                cov = 0.245
            else:
                cov = 0
            covList.append(cov)
        invCovMat.append(covList)
    return  invCovMat

def inverseMatrix(invCovMat):
    CovMat = matrix(invCovMat).I
    return CovMat

def convCoordinate(Vec):
    Vec = [ e for e in Vec for repetitions in range(len(Vec)) ]
    return Vec

def drawGraph(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r', marker='o')
    def fun(x, y):
        return (1 / math.sqrt(2 * math.pi)) * math.exp((-4) * (x**2 + y**2))
    x = y = np.arange(-3.0, 3.0, 0.05)
    X1, Y1 = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X1), np.ravel(Y1))])
    Z1 = zs.reshape(X1.shape)

    ax.plot_surface(X1, Y1, Z1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def isSymmetrix(mat):
    return(mat.transpose() == mat).all()
Sample = uniformSample()
PrMat = calProbInMatrix(Sample)
AdMat = calAdMatrix(PrMat)
invCovMat = calInverseCovMatrix(AdMat)
CovMat = inverseMatrix(invCovMat)
np.save("covariance_Matrix", CovMat)


def drawAdGraph(X, Admat):
    # fig = plt.figure()
    G = nx.Graph()
    rows, cols = np.where(AdMat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr)
    plt.title('Adjacency Matrix')
    plt.show()

def drawImage(mat):
    plt.imshow(np.asarray(mat), interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5, 10.5, 0.5, 10.5))
    plt.pcolor(np.asarray(mat))
    plt.colorbar()
    plt.show()

def drawAdGraph(AdMat, invCovMat,CovMat):

    # graph
    fig = plt.figure()
    plt.subplot(131, aspect='equal')
    rows, cols = np.where(AdMat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr)
    plt.title('Adjacency Matrix')

    # Cov
    plt.subplot(132, aspect='equal')
    im1=plt.imshow(np.asarray(CovMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Adjacency Matrix')

    # Inverse
    plt.subplot(133, aspect='equal')
    im2 =plt.imshow(np.asarray(invCovMat), interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('Adjacency Matrix')

    plt.show()





print np.asarray(CovMat).shape, np.asarray(invCovMat).shape
Z = [z for sublist in AdMat for z in sublist]
drawAdGraph(AdMat, invCovMat,CovMat)
