import numpy
import urllib

print "Reading data..."
dataFile = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"', '').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
print "done"

X_train = X[:int(len(X) / 3)]
y_train = y[:int(len(y) / 3)]

X_mean=numpy.mean(X_train,axis=0)
diff=X_train-X_mean
re=sum(sum(numpy.multiply(diff,diff)).T)
print re