import gzip
import numpy as np
from collections import defaultdict

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

print "Reading data..."
data = [l for l in readGz("train.json.gz")]
print "Done."

data_train=data[:100000]
data_validation=data[100000:]
data_train_valid=[datum for datum in data_train if datum['helpful']['outOf']!=0]

def feature(datum):
    feat=[1]
    word=datum['reviewText'].split()
    word_count=len(word)
    feat.append(word_count)
    feat.append(datum['rating'])
    return feat

X_train=[feature(d) for d in data_train_valid]
y_train=[datum['helpful']['nHelpful']*1.0/datum['helpful']['outOf'] for datum in data_train_valid]
##fit a lst
theta,residuals,rank,s = np.linalg.lstsq(X_train,y_train)
print 'theta=',theta

X_validation=[feature(d) for d in data_validation]
theta=np.matrix(theta)
X_validation=np.matrix(X_validation)
rate_predict=X_validation*theta.T

outOf_validation=[datum['helpful']['outOf'] for datum in data_validation]
outOf_validation=np.matrix(outOf_validation)
y_predict=np.multiply(outOf_validation.T,rate_predict)

y_validation=[datum['helpful']['nHelpful'] for datum in data_validation]
y_predict=y_predict.T.tolist()[0]

diff=[y_validation[i]-y_predict[i] for i in range(len(y_predict))]
error=[abs(i) for i in diff]
MAE=sum(error)/len(y_predict)
print 'MAE=',MAE
