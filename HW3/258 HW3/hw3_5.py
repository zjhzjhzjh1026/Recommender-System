import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

print "Reading data..."
data = [l for l in readGz("train.json.gz")]
print "Done."
data_train=data[:100000]
data_validation=data[100000:]

rating_train=[datum['rating'] for datum in data_train]
alpha=sum(rating_train)/len(data_train)

rating_validation=[datum['rating'] for datum in data_validation]
error=[(t-alpha)**2 for t in rating_validation]
MSE=sum(error)/len(rating_validation)
print "MSE=",MSE