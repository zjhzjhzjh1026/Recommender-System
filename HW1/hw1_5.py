import numpy
import scipy.optimize
import random
import urllib
import csv
from sklearn import svm

print "Reading data..."
url=urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
data=list(csv.DictReader(url,delimiter=";"))
print "Done."

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

clf = svm.SVC(C=0.8)
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

accuracy_train=sum([z[0]==z[1] for z in zip(train_predictions,y_train)])*1.0/len(train_predictions)
accuracy_test=sum([z[0]==z[1] for z in zip(test_predictions,y_test)])*1.0/len(test_predictions)


print "The train accuracy is: ",accuracy_train
print "The test accuracy is: ",accuracy_test
