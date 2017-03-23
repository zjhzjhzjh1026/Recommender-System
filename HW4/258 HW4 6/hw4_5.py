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

wordList = ['foam','smell','banana','lactic','tart']
wordCount = defaultdict(int)
punctuation = set(string.punctuation)

for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  w = r.split()
  for word in wordList:
    if word in w:
      wordCount[word] += 1

freq = [math.log10(len(data) * 1.0 /wordCount[word]) for word in wordList]
freq

d=data[0]
wordCount2 = defaultdict(int)
r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
w = r.split()
for word in w:
    if word in wordList:
      wordCount2[word] += 1

tfidf=[freq[i]*wordCount2[wordList[i]] for i in range(len(wordList))]
tfidf