import numpy
import urllib
import scipy.optimize
import random
import matplotlib.pyplot as plt
from math import exp
from math import log

random.seed(0)


def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)


print "Reading data..."
dataFile = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"', '').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[1.0] + [float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
print "done"


def inner(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k] * theta[k]
    # for debugging
    # print "ll =", loglikelihood
    return -loglikelihood


# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
    dl = [0] * len(theta)
    for i in range(len(X)):
        logit = inner(X[i], theta)
        for k in range(len(theta)):
            dl[k] += X[i][k] * (1 - sigmoid(logit))
            if not y[i]:
                dl[k] -= X[i][k]
    for k in range(len(theta)):
        dl[k] -= lam * 2 * theta[k]
    return numpy.array([-x for x in dl])


X_train = X[:int(len(X) / 3)]
y_train = y[:int(len(y) / 3)]
X_validate = X[int(len(X) / 3):int(2 * len(X) / 3)]
y_validate = y[int(len(y) / 3):int(2 * len(y) / 3)]
X_test = X[int(2 * len(X) / 3):]
y_test = y[int(2 * len(X) / 3):]


##################################################
# Train                                          #
##################################################

def train(lam):
    theta, _, _ = scipy.optimize.fmin_l_bfgs_b(f, [0] * len(X[0]), fprime, pgtol=10, args=(X_train, y_train, lam))
    return theta


##################################################
# Predict                                        #
##################################################

def performance3(theta,numR):

    scores_test = [inner(theta, x) for x in X_test]
    combine = [[scores_test[i]] + [y_test[i]] for i in range(0, len(y_test))]
    combine.sort(key=lambda x: x[0], reverse=True)
    total = sum(l[1]==True for l in combine)
    rel = sum(combine[i][1]==True for i in range(0,numR))
    precision = rel * 1.0 / numR
    recall = rel *1.0 / total

    return precision,recall


lam = 0.01
theta = train(lam)
result_pre = []
result_rec = []
for numReturn in range(1,len(y_test)+1):
    precision,recall = performance3(theta,numReturn)
    result_pre.append(precision)
    result_rec.append(recall)
plt.plot(result_rec,result_pre)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall')
plt.show()
