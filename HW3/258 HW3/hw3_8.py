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

allRatings = []
userRating = defaultdict(list)
itemRating = defaultdict(list)
for l in data_train:
    user,item = l['reviewerID'],l['itemID']
    allRatings.append(l['rating'])
    userRating[user].append(l['rating'])
    itemRating[item].append(l['rating'])
alpha = sum(allRatings) / len(allRatings)
userAverage = {}
itemAverage = {}
lam=7
for u in userRating:
    userAverage[u] = sum(userRating[u]) / len(userRating[u])
for i in itemRating:
    itemAverage[i] = sum(itemRating[i]) / len(itemRating[i])
k=0
mse1,mse2,mse3,mse4,mse5 = 0,1000,0,1000,0
while(abs(mse2-mse1)+abs(mse3-mse2)+abs(mse4-mse3)+abs(mse5-mse4)>0.00001):
    s1=[datum['rating']-userAverage[datum['reviewerID']]-itemAverage[datum['itemID']] for datum in data_train]
    alpha=sum(s1)/len(data_train)
    userRating = defaultdict(list)
    for l in data_train:
        userRating[l['reviewerID']].append(l['rating']-alpha-itemAverage[l['itemID']])
    for u in userRating:
        userAverage[u] = sum(userRating[u]) / (len(userRating[u])+lam)
    itemRating = defaultdict(list)
    for l in data_train:
        itemRating[l['itemID']].append(l['rating']-alpha-userAverage[l['reviewerID']])
    for u in itemRating:
        itemAverage[u] = sum(itemRating[u]) / (len(itemRating[u])+lam)
    k = k + 1

    rating_validation = []
    for datum in data_validation:
        predict = alpha
        if(datum['itemID'] in itemAverage):
            predict = predict + itemAverage[datum['itemID']]
        if(datum['reviewerID'] in userAverage):
            predict = predict + userAverage[datum['reviewerID']]
        rating_validation.append(predict)
    error = [(rating_validation[i]-data_validation[i]['rating'])**2 for i in range(len(data_validation))]
    mse=sum(error)/len(data_validation)
    #print k," MSE= ",mse
    mse1, mse2, mse3, mse4, mse5 = mse, mse1, mse2, mse3, mse4
print "Final MSE= ",mse

predictions = open("output_rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  predict = alpha
  if (i in itemAverage):
      predict = predict + itemAverage[i]
  if (u in userAverage):
      predict = predict + userAverage[u]
  predictions.write(u + '-' + i + ',' + str(predict) + '\n')

predictions.close()