import numpy
import urllib    #read data from web
import scipy.optimize    #optimize tool
import random

def parseData(fname):      #a record per json
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"
###use iterator produced by yield to create list, each record is consist of several key/value pairs
def feature(datum):
  feat = [1]
  return feat

X = [feature(d) for d in data]  #[[1], [1], [1], [1], [1], [1], [1], [1], [1], [1] (repeat for 50000 times)]
y = [d['review/overall'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)   #最小二乘法拟合线性方程

### Convince ourselves that basic linear algebra operations yield the same answer ###

X = numpy.matrix(X)     #列向量now
y = numpy.matrix(y)
numpy.linalg.inv(X.T * X) * X.T * y.T

### Do older people rate beer more highly? ###

data2 = [d for d in data if d.has_key('user/ageInSeconds')]  #选出50000条中有年龄字段的记录

def feature(datum):      # 相比上面形成的是[1,age]对
  feat = [1]
  feat.append(datum['user/ageInSeconds'])
  return feat

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### How much do women prefer beer over men? ###

data2 = [d for d in data if d.has_key('user/gender')]

def feature(datum):
  feat = [1]
  if datum['user/gender'] == "Male":
    feat.append(0)
  else:
    feat.append(1)
  return feat

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### Gradient descent ###

# Objective
def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  print "offset =", diffSqReg.flatten().tolist()
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  print "gradient =", numpy.array(res.flatten().tolist()[0])
  return numpy.array(res.flatten().tolist()[0])

scipy.optimize.fmin_l_bfgs_b(f, [0,0], fprime, args = (X, y, 0.1))    ##Minimize a function func using the L-BFGS-B algorithm.

### Random features ###

def feature(datum):
  return [random.random() for x in range(30)]

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
