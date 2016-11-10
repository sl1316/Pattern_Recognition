import numpy as np
import math
from numpy import matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def uniformSample():
    return np.random.rand(20, 2)
def calProbInMatrix(sample):
    def calProb(x, y):
        def dist(a, b):
            # print (a - b)
            # print (a - b)**2
            # print np.sum((a-b)**2)
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
    #print np.array(ProbMat).shape
    return ProbMat
def calAdMatrix(ProbMat):
    #S = np.zeros((ProbMat.shape[0], ProbMat.shape[1]))
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
#print PrMat
#print AdMat
#print Sample
def drawAdGraph(X, Admat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1],color = 'r')
    for i in range(len(AdMat)):
        for j in range(len(AdMat[0])):
            if(AdMat[i][j] == 1):
                plt.plot([X[i][0], X[j][0]], [X[i][1], X[j][1]])

    plt.show()

def drawImage(mat):
    #plt.imshow(np.asarray(mat), interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5, 10.5, 0.5, 10.5))
    plt.pcolor(np.asarray(mat))
    plt.colorbar()
    plt.show()

drawImage(invCovMat)
drawImage(CovMat)
print np.asarray(CovMat).shape, np.asarray(invCovMat).shape
Z = [z for sublist in AdMat for z in sublist]
# X = convCoordinate(Sample[:, 0])
# Y = convCoordinate(Sample[:, 1])
#print len(X), len(Y), len(Z)
drawAdGraph(Sample, AdMat)
#drawGraph(X, Y, Z)
#print CovMat
#print invCovMat * CovMat











# #P = np.ones((4, 5)) * 0.2
# P =[[ 0.4,  0.2,  0.2,  0.2 , 0.2]
#  ,[ 0.2, 0.2 , 0.2 , 0.2 , 0.2]
#  ,[ 0.2 , 0.2 , 0.2 , 0.2 , 0.2]
#  ,[ 0.2 , 0.2  ,0.2 , 0.2  ,0.2]]
# S = np.zeros((4, 5))
# print P
# print S
# for i in range(10000):
#     S += (P > np.random.rand(4, 5)).astype(int)
#
# print S          # each element should be approximately 20
# print S.mean()   # the average  should be approximately 20, too
