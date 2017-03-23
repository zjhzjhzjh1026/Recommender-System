import numpy
import urllib
import scipy.optimize
import random
import csv
from math import exp
from math import log

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)

print "Reading data..."
url=urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
data=list(csv.DictReader(url,delimiter=";"))
print "done"

keys=["alcohol","chlorides","citric acid","density","fixed acidity","free sulfur dioxide","pH"
    ,"residual sugar","sulphates","total sulfur dioxide","volatile acidity"]
def feature(datum):
    feat=[]
    for key in keys:
        feat.append(eval(datum[key]))
    return feat
X=[feature(d) for d in data]
y=[eval(d["quality"])>5 for d in data]

l=len(data)     #split the data into 2 groups
X_train = X[:l/2]
y_train = y[:l/2]

X_test = X[l/2:]
y_test = y[l/2:]

def inner(x,y):
    return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood

def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k]*theta[k]
    #print "ll =", loglikelihood
    return -loglikelihood

def fprime(theta, X, y, lam):
    dl = [0.0]*len(theta)
    for j in range(len(theta)):
        for i in range(len(X)):
            logit = inner(X[i], theta)
            dl[j] += X[i][j] * (1 - sigmoid(logit))
            if not y[i]:
                dl[j] -= X[i][j]
        dl[j] -= 2 * lam * theta[j]
    # Negate the return value since we're doing gradient *ascent*
    #print "dll =", dl
    return numpy.array([-x for x in dl])


# Use a library function to run gradient descent (or you can implement yourself!)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print "Final log likelihood =", -l

ans=[int(tmp>0) for tmp in numpy.matrix(X_test)*numpy.matrix(theta).T]
accuracy_test=sum([z[0]==z[1] for z in zip(ans,y_test)])*1.0/len(ans)

print "Accuracy = ", accuracy_test # Compute the accuracy