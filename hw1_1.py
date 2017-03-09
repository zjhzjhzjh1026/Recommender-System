import numpy
import scipy.optimize
import random
import urllib

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "Done."

def feature(datum):
    feat=[1]
    feat.append(datum['review/timeStruct']['year'])
    return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X,y)
print theta
MSE=residuals/len(y)
print MSE

tmp=numpy.matrix(X)
minV=tmp[:,1].min()
maxV=tmp[:,1].max()

def feature2(datum):
    feat=[1]
    tmp=datum['review/timeStruct']['year']
    l=[0 for x in range(13)]
    if tmp!=maxV:
        l[tmp-minV]=1
    feat.extend(l)
    return feat

X = [feature2(d) for d in data]
y = [d['review/overall'] for d in data]

theta,residuals,rank,s = numpy.linalg.lstsq(X,y)
MSE2=residuals/len(y)
print MSE2