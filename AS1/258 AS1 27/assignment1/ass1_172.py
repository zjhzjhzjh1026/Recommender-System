import gzip
import numpy
from collections import defaultdict
import scipy.optimize
import random
import string

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

print "Reading data..."
data = [l for l in readGz("train.json.gz")]
print "Done."

random.seed(10)
random.shuffle(data)
data_train=data[:100000]
data_validation=data[100000:]
data_train_valid=[datum for datum in data_train if datum['helpful']['outOf']!=0]

def parse_string(str):
    cnt_ques , cnt_up , cnt_punc , cnt_char  = 0 , 0 , 0 , 0
    for c in str:
        if c in ['?','!']:
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

def outOf_plank(num):
    if num<1:
        return 0
    elif num<5:
        return 1
    elif num<10:
        return 2
    elif num<20:
        return 3
    elif num<50:
        return 4
    else:
        return 5

def feature(datum):
    feat=[1]
    ana=parse_string(datum['reviewText'])
    for i in ana:
        feat.append(i)
    feat.append(outOf_plank(datum['helpful']['outOf']))
    feat.append(outOf_plank(datum['helpful']['outOf'])**2)
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

theta=scipy.optimize.fmin_l_bfgs_b(f, [0]*9, fprime, args = (X_train, y_train, 0))[0]

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

data_test = [l for l in readGz("test_Helpful.json.gz")]
X_test=[feature(d) for d in data_test]
theta=numpy.matrix(theta)
X_test=numpy.matrix(X_test)
test_predict=X_test*theta.T

outOf_test=[datum['helpful']['outOf'] for datum in data_test]
outOf_test=numpy.matrix(outOf_test)
y_predict=numpy.multiply(outOf_test.T,test_predict)
y_predict=y_predict.T.tolist()[0]


predictions = open("predictions_Helpful.txt", 'w')
cur=0
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,prediction = l.strip().split('-')
  predictions.write(u + '-' + i + '-' + prediction + ',' + str(round(y_predict[cur])) + '\n')
  cur=cur+1
predictions.close()