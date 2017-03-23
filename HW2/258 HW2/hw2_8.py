import numpy
import urllib
from sklearn.decomposition import PCA
import scipy.optimize
import random
import csv
from math import exp
from math import log
import matplotlib.pyplot as plt

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)


print "Reading data..."
dataFile = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
header = dataFile.readline()
fields = header.strip().replace('"', '').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1]  for l in lines]
print "done"

X_train = X[:int(len(X) / 3)]
y_train = y[:int(len(y) / 3)]
X_test = X[int(2 * len(X) / 3):]
y_test = y[int(2 * len(X) / 3):]

def inner(x,y):
    return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

# Objective
def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  #print "offset =", diffSqReg.flatten().tolist()
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  #print "gradient =", numpy.array(res.flatten().tolist()[0])
  return numpy.array(res.flatten().tolist()[0])

train_mse=[]
test_mse=[]

for dim in range(1,12):
    pca = PCA(n_components=dim)
    X_PCA = pca.fit_transform(X_train).tolist()
    X_PCA = [[1] + elem for elem in X_PCA]
    theta, l, info = scipy.optimize.fmin_l_bfgs_b(f, [0]*(dim+1), fprime, args = (X_PCA, y_train, 0))  #lambda
    #print "Final log likelihood =", -l
    X_test_PCA = pca.transform(X_test).tolist()
    X_test_PCA = [[1] + elem for elem in X_test_PCA]
    X_test_PCA=numpy.matrix(X_test_PCA)
    theta=numpy.matrix(theta)
    y_test=numpy.matrix(y_test)
    y=X_test_PCA*theta.T
    k=y-y_test.T
    mse2=(k.T*k/len(y)).tolist()[0][0]
    test_mse.append(mse2)

    X_PCA = numpy.matrix(X_PCA)
    y_train = numpy.matrix(y_train)
    y = X_PCA * theta.T
    k = y - y_train.T
    mse1 = (k.T * k / len(y)).tolist()[0][0]
    train_mse.append(mse1)
XX=range(1,12)
plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
plt.sca(ax1)
plt.plot(XX,train_mse)
plt.title('Training Set MSE')
plt.xlabel('Dimension')
plt.ylabel('MSE')
plt.sca(ax2)
plt.plot(XX,test_mse)
plt.xlabel('Dimension')
plt.ylabel('MSE')
plt.title('Test Set MSE')
plt.show()