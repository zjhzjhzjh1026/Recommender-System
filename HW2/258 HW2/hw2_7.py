import numpy
import urllib
from sklearn.decomposition import PCA

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
y = [l[-1] > 5 for l in lines]
print "done"

X_train = X[:int(len(X) / 3)]
y_train = y[:int(len(y) / 3)]
X_validate = X[int(len(X) / 3):int(2 * len(X) / 3)]
y_validate = y[int(len(y) / 3):int(2 * len(y) / 3)]
X_test = X[int(2 * len(X) / 3):]
y_test = y[int(2 * len(X) / 3):]

pca = PCA(n_components=4)
pca.fit(X_train)


X_PCA=pca.fit_transform(X_train)
x_rec2=pca.inverse_transform(X_PCA)
x_rec2=numpy.matrix(x_rec2)
error2=sum(sum(numpy.multiply(x_rec2-X_train,x_rec2-X_train)).T).tolist()[0][0]
print error2

print pca.noise_variance_*len(x_rec2)   ##This single command enough