import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)
print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))[:5000]
print "done"

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  w = r.split()
  for i in range(len(w)-1):
    wordCount[w[i]+' '+w[i+1]] += 1
  for tmp in w:
    wordCount[tmp] += 1
len(wordCount)

counts = [(wordCount[w] , w) for w in wordCount]
counts.sort()
counts.reverse()
# counts
words = [x[1] for x in counts[:1000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  w = r.split()
  for i in range(len(w)-1):
    bitemp=w[i]+' '+w[i+1]
    if bitemp in words:
      feat[wordId[bitemp]] += 1
  for tmp in w:
    if tmp in words:
      feat[wordId[tmp]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)

diff = predictions-y
mse = sum([t**2 for t in diff])/len(diff)
mse

weight = [(theta[i] , words[i]) for i in range(len(words))]
weight.sort()
weight

weight.reverse()
weight