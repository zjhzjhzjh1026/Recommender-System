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

data_test = [l for l in readGz("test_Helpful.json.gz")]
X_test=[feature(d) for d in data_test]
theta=np.matrix(theta)
X_test=np.matrix(X_test)
test_predict=X_test*theta.T

outOf_test=[datum['helpful']['outOf'] for datum in data_test]
outOf_test=np.matrix(outOf_test)
y_predict=np.multiply(outOf_test.T,test_predict)
y_predict=y_predict.T.tolist()[0]

predictions = open("predictions_Helpful.txt", 'w')
cur=0
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,prediction = l.strip().split('-')
  predictions.write(u + '-' + i + '-' + prediction + ',' + str(y_predict[cur]) + '\n')
  cur=cur+1
predictions.close()