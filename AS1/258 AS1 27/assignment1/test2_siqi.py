import gzip
import numpy
import string
import scipy.optimize
from collections import defaultdict
import random


#random.seed(0)
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

data = [l for l in readGz("train.json.gz")]
random.shuffle(data)

def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T    # T means transpose
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  return numpy.array(res.flatten().tolist()[0])

def  numofwords(s):
    return len([w for w in s.split(' ') if w])

def numofpunc(s):
    num_punc = 0
    for c in s:
        if c in [',','.','?','!']:
             num_punc += 1
    return num_punc

def outoflevel(a):
  level = 0.5
  if (a >= 1) & (a < 3):
    level = 1
  if (a >= 3) & (a < 5):
    level = 2
  if (a >= 5) & (a < 8):
    level = 3
  if (a >= 8) & (a < 18):
    level = 4
  if (a >= 18) & (a < 25):
    level = 5
  if (a >= 25) & (a < 40):
    level = 6
  if a >= 40:
    level = 7
  return level

def isWomen(c):
  gender = 0
  for i in range(len(c)):
    for j in range(len(c[i])):
        if c[i][j] in ['Women']:
          gender = 1
  return gender

def isGirls(c):
  gender = 0
  for i in range(len(c)):
    for j in range(len(c[i])):
        if c[i][j] in ['Girls']:
          gender = 1
  return gender

def isMen(c):
  gender = 0
  for i in range(len(c)):
    for j in range(len(c[i])):
        if c[i][j] in ['Men']:
          gender = 1
  return gender

def isBoys(c):
  gender = 0
  for i in range(len(c)):
    for j in range(len(c[i])):
        if c[i][j] in ['Boys']:
          gender = 1
  return gender

def isBaby(c):
  gender = 0
  for i in range(len(c)):
    for j in range(len(c[i])):
        if c[i][j] in ['Baby']:
          gender = 1
  return gender

def findgender(c):
  gender = 1
  for i in range(len(c)):
     for j in range(len(c[i])):
        if c[i][j] in ['Women', 'Girls']:
            gender = 0
        if c[i][j] in ['Men', 'Boys']:
            gender = 2
  return gender

def finduseful(s):
    ws = [w for w in s.split(' ') if w]
    num_use = 0
    for w in ws:
        if w in ['!','love','perfect','awesome','amazing','great','best','loves','perfectly','excellent','good','bad','disappointed','not','boring','poor','return','terrible','horrible','worst','waste','never','again']:
            num_use += 1
    return num_use

def outofVect(a):
    vect = [0]*10
    if a < 10:
        vect[a] = 1
        vect.append(0)
    else:
        vect.append(a)
    return vect

def feature(datum):
  feat = [1]
  feat.append(finduseful(datum['reviewText']))
  feat.append(finduseful(datum['summary']))
  feat.append(datum['rating'])
  feat.append(datum['rating']**2)
  feat.extend(outofVect(datum['helpful']['outOf']))
  feat.append(outoflevel(datum['helpful']['outOf']))
  feat.append(outoflevel(datum['helpful']['outOf'])**2)
#  feat.append(findgender(datum['categories']))
  feat.append(isBaby(datum['categories']))
  feat.append(isBoys(datum['categories']))
  feat.append(isMen(datum['categories']))
  feat.append(isWomen(datum['categories']))
  feat.append(isGirls(datum['categories']))

  return feat

allRatio = []
userRatings = defaultdict(list)
itemRatings = defaultdict(list)
userRatio = defaultdict(list)
itemRatio = defaultdict(list)
for l in readGz("train.json.gz"):
  user, item = l['reviewerID'], l['itemID']
  itemRatings[item].append(l['rating'])
  userRatings[user].append(l['rating'])
  if l['helpful']['outOf'] != 0:
    allRatio.append(l['helpful']['nHelpful'] * 1.0 / l['helpful']['outOf'])
    userRatio[user].append(l['helpful']['nHelpful'] * 1.0 / l['helpful']['outOf'])
    itemRatio[item].append(l['helpful']['nHelpful'] * 1.0 / l['helpful']['outOf'])

for item in itemRatings:
  itemRatings[item] = sum(itemRatings[item]) / len(itemRatings[item])

for user in userRatings:
  userRatings[user] = sum(userRatings[user]) / len(userRatings[user])

for user in userRatio:
  userRatio[user] = sum(userRatio[user]) / len(userRatio[user])

for item in itemRatio:
  itemRatio[item] = sum(itemRatio[item]) / len(itemRatio[item])

globalAverage = sum(allRatio) / len(allRatio)

random.shuffle(data)
Traindata = data[:100000]
Validatedata = data[100000:]

XTrain = [feature(l) for l in Traindata if l['helpful']['outOf'] != 0]
yTrain = [l['helpful']['nHelpful']*1.0/l['helpful']['outOf'] for l in Traindata if l['helpful']['outOf'] != 0]
Trainoutof = [l['helpful']['outOf'] for l in Traindata if l['helpful']['outOf'] != 0]
TrainnHelpful = [l['helpful']['nHelpful'] for l in Traindata if l['helpful']['outOf'] != 0]

XValidate = [feature(l) for l in Validatedata]
Validateoutof = [l['helpful']['outOf'] for l in Validatedata]
ValidatenHelpful = [l['helpful']['nHelpful'] for l in Validatedata]

Testdata = [l for l in readGz("test_Helpful.json.gz")]
XTest = [feature(l) for l in Testdata]

theta = scipy.optimize.fmin_l_bfgs_b(f, [0]*23, fprime, args = (XTrain, yTrain, 0))[0]

XTest = numpy.matrix(XTest)
theta = numpy.matrix(theta)
yTest_prediction = theta * XTest.T

XTrain = numpy.matrix(XTrain)
theta = numpy.matrix(theta)
yTrain_prediction = numpy.multiply(theta * XTrain.T, Trainoutof)

XValidate = numpy.matrix(XValidate)
theta = numpy.matrix(theta)
yValidate_prediction = numpy.multiply(theta * XValidate.T, Validateoutof).tolist()[0]

ValidateMAE = sum([abs(round(a) - b) for (a, b) in zip(yValidate_prediction, ValidatenHelpful)]) / len(ValidatenHelpful)

print 'Problem 3 ValidateMAE = ' + str(ValidateMAE)
print 'Problem 4 test prediction is in yTest_prediction'