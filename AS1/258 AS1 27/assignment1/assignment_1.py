import gzip
import numpy
from collections import defaultdict
import scipy.optimize
import random

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

print "Reading data..."
data = [l for l in readGz("train.json.gz")]
print "Done."

random.seed(1)
data.shuffle()
data_train=data[:100000]
data_validation=data[100000:]
data_train_valid=[datum for datum in data_train if datum['helpful']['outOf']!=0]

def parse_string(str):
    cnt_ques , cnt_up , cnt_punc , cnt_char  = 0 , 0 , 0 , 0
    for c in str:
        if c in ['?']:
            cnt_ques += 1
        if c in string.uppercase:
            cnt_up+= 1
        if c in string.punctuation:
            cnt_punc += 1
        cnt_char += 1
    if cnt_char == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return cnt_ques * 1.0 / cnt_char, cnt_up * 1.0 / cnt_char, cnt_ques * 1.0 / cnt_char, \
           len(str.strip().split()), cnt_char



def feature(datum):
    feat=[1]
    feat.append(parse_string(datum['reviewText']))
    feat.append(datum['helpful']['outOf'])
    feat.append(datum['helpful']['outOf']**2)
    feat.append(datum['rating'])
    return feat

X_train=[feature(d) for d in data_train_valid]
y_train=[datum['helpful']['nHelpful']*1.0/datum['helpful']['outOf'] for datum in data_train_valid]
##fit a lst
#theta,residuals,rank,s = numpy.linalg.lstsq(X_train,y_train)

def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  #print "offset =", diffSqReg.flatten().tolist()
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  #print "gradient =", numpy.array(res.flatten().tolist()[0])
  return numpy.array(res.flatten().tolist()[0])

theta=scipy.optimize.fmin_l_bfgs_b(f, [0]*9, fprime, args = (X_train, y_train, 0.1))[0]

X_validation=[feature(d) for d in data_validation]
theta=numpy.matrix(theta)
X_validation=numpy.matrix(X_validation)
rate_predict=X_validation*theta.T

outOf_validation=[datum['helpful']['outOf'] for datum in data_validation]
outOf_validation=numpy.matrix(outOf_validation)
y_predict=numpy.multiply(outOf_validation.T,rate_predict)

y_validation=[datum['helpful']['nHelpful'] for datum in data_validation]
y_predict=y_predict.T.tolist()[0]
for i in range(len(y_predict)):
    if y_predict[i]>data_validation[i]['helpful']['outOf']:
        y_predict[i]=data_validation[i]['helpful']['outOf']
diff=[y_validation[i]-y_predict[i] for i in range(len(y_predict))]
error=[abs(i) for i in diff]
MAE=sum(error)/len(y_predict)
print 'MAE=',MAE
