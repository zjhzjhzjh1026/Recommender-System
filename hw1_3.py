import numpy
import scipy.optimize
import random
import urllib
import csv

print "Reading data..."
url=urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
data=list(csv.DictReader(url,delimiter=";"))
print "Done."

l=len(data)     #split the data into 2 groups
train=data[0:l/2]
test=data[l/2:]

keys=["alcohol","chlorides","citric acid","density","fixed acidity","free sulfur dioxide","pH"
    ,"residual sugar","sulphates","total sulfur dioxide","volatile acidity"]
def feature(datum):
    feat=[1]
    for key in keys:
        feat.append(eval(datum[key]))
    return feat
X=[feature(d) for d in train]
y=[eval(d["quality"]) for d in train]
theta,residuals,rank,s = numpy.linalg.lstsq(X,y)
print "Theta:",theta
print "Training Set MSE:",residuals[0]/len(train)

X_t=[feature(d) for d in test]
y_t=[eval(d["quality"]) for d in test]

X_t=numpy.matrix(X_t)
theta=numpy.matrix(theta)
y_t=numpy.matrix(y_t)
diff=theta*X_t.T-y_t
mse=diff*diff.T
k = mse.tolist()[0]
print "Test Set MSE:", k[0] / len(test)