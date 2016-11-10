import numpy as np
import matplotlib.pyplot as plt
from numpy import matrix
from scipy.stats import multivariate_normal
def drawImage(mat,figureNum,title):
    plt.subplot(figureNum)
    #plt.imshow(np.asarray(mat), interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5, 10.5, 0.5, 10.5))
    plt.pcolor(np.asarray(mat))
    plt.title(title)
    plt.colorbar()

def inverseMatrix(invCovMat):
    CovMat = matrix(invCovMat).I
    return CovMat
# def normMat(difMat,origMat):
#     nMat = []
#     for i in range(len(difMat)):
#         tmp = []
#         for j in range(len(difMat[0])):
#             if origMat[i][j] != 0:
#                 tmp.append(difMat[i][j] / origMat[i][j])
#             # else:
#             #     tmp.append()
#         nMat.append(tmp)
#     return nMat
u = 1.5
cov = np.load("covariance_Matrix.npy").astype(float)
#print cov
mean = [u for i in range(20)]
#cov = [[1, 0], [0, 100]]
sample = np.random.multivariate_normal(mean, cov, 1000)
np.save("sample", sample)
covMat = np.cov(sample.T)
precisionMat = inverseMatrix(covMat)
drawImage(cov,221, "Model Covariance Matrix")
drawImage(covMat,222, "Sample Covariance Matrix")
drawImage(precisionMat, 223, "Sample Precision Matrix")
plt.tight_layout()
plt.show()
#plt.plot(x, y, 'x')
#plt.axis('equal')
#plt.show()