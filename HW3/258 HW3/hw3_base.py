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

nHelpful=[t['helpful']['nHelpful'] for t in data_train]
outOf=[t['helpful']['outOf'] for t in data_train]
total_nHelpful=sum(nHelpful)
total_outOf=sum(outOf)
alpha=total_nHelpful*1.0/total_outOf
print "alpha=",alpha

validation_outOf=[t['helpful']['outOf'] for t in data_validation]
validation_nHelpful=[t['helpful']['nHelpful'] for t in data_validation]
diff=[validation_outOf[i]*alpha-validation_nHelpful[i] for i in range(len(validation_outOf))]
error=[abs(i) for i in diff]
MAE=sum(error)/len(validation_outOf)
print "MAE=",MAE