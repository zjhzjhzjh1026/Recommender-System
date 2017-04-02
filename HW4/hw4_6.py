import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import math

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
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

wordList = [x[1] for x in counts[:1000]]

wordCount1 = defaultdict(int)
punctuation = set(string.punctuation)

for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  w = r.split()
  for word in wordList:
    if word in w:
      wordCount1[word] += 1

freq = [math.log10(len(data) * 1.0 /wordCount1[word]) for word in wordList]
d=data[0]
wordCount2 = defaultdict(int)
r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
w = r.split()
for word in w:
    if word in wordList:
      wordCount2[word] += 1
tfidf1=[freq[i]*wordCount2[wordList[i]] for i in range(len(wordList))]

d=data[1]
wordCount2 = defaultdict(int)
r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
w = r.split()
for word in w:
    if word in wordList:
      wordCount2[word] += 1
tfidf2=[freq[i]*wordCount2[wordList[i]] for i in range(len(wordList))]
print 1-scipy.spatial.distance.cosine(tfidf1,tfidf2)