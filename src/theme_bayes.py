from nlpa_wrap import NLModel
from nlpcc_parser import SMUM_Data
from pandas import merge
from time import time
from pickle import load, dump
from nlpcc_parser import SMUM_Data
from util import prob_normalize
from collections import defaultdict
from build import DataSet
from nlpa_wrap import NLModel

data = SMUM_Data()
data.load()
data.filter()
nlm = NLModel()
nlm.loadStopwords()

tweet_with_profile = merge(data.tweets, data.profile)
male = tweet_with_profile[tweet_with_profile.gender == 'm'].tweet
female = tweet_with_profile[tweet_with_profile.gender == 'f'].tweet
print "<%d male record, %d female record>"%(len(male), len(female))

male = nlm.splitTexts(male, 'tweets_male')
female = nlm.splitTexts(female, 'tweets_female')

nlm.updateDict(male)
nlm.updateDict(female)
nlm.filterDict(5)
mg = male + female

male = nlm.computeCorpus(male)
female = nlm.computeCorpus(female)
male_tfidf, _ = NLModel.computeTfidf(male, context=female+male, normalize=False)
female_tfidf, _ = NLModel.computeTfidf(female, context=male+female, normalize=False)

mg = nlm.computeCorpus(mg)
tfidf, tfidf_model = NLModel.computeTfidf(mg, context=mg, normalize=False)

lsi = nlm.extractLSI(tfidf, 200)

def top(x, n = 4.0):
    if x == []:
        return []
    x.sort(key=lambda x:x[1], reverse=True)
    for i, v in enumerate(x):
        if v[1] < n:
            if i == 0:
                return []
            return zip(*x[:i])[0]
    return zip(*x)[0]

# filter
lst = []
for i in range(len(mg)):
    print '%.2f'%(i/float(len(mg))*100)
    lst.append(top(lsi[tfidf[i]]))

##############################################
# learning topic postier probility
dct = defaultdict(int)  # Male
for i in range(len(male)):
    for j in lst[i]:
        dct[j] += 1
prob_m = prob_normalize(dct.items())
for k, v in prob_m:
    dct[k] = v

dct_f = defaultdict(int) # Female
for i in range(len(male), len(mg)):
    for j in lst[i]:
        dct_f[j] += 1
prob_m = prob_normalize(dct_f.items())
for k, v in prob_m:
    dct_f[k] = v
##############################################

##############################################
# learning tag postier probility
dtm = defaultdict(int)
tgm = list(data_set.train_male.tag.tag)
for t in tgm:
    dtm[t] += 1
prob_m = prob_normalize(dtm.items())
for k, v in prob_m:
    dtm[k] = v

dtf = defaultdict(int)
tgf = list(data_set.train_female.tag.tag)
for t in tgf:
    dtf[t] += 1
prob_f = prob_normalize(dtf.items())
for k, v in prob_f:
    dtf[k] = v
###############################################

data_set = DataSet(data)
data_set.build(0.8)

tweet = data_set.test_female.tweet
tag = data_set.test_female.tag
ids = set(tweet.id)

corr = 0
for user in ids:
    tweets = list(tweet[tweet.id == user].tweet)
    tweets = nlm.splitTexts(tweets)
    _corpus = nlm.computeCorpus(tweets)
    _tfidf = [tfidf_model[x] for x in _corpus]
    _lsi = [lsi[x] for x in _tfidf]
    _lsi = [top(x) for x in _lsi]
    ttag = list(tag[tag.id == user].tag)
    mm = 0
    ff = 0
    for x in _lsi:
        pm = 0.5
        pf = 0.5
        for y in x:
            if y in dct and y in dct_f:
                pm *= dct[y]
                pf *= dct_f[y]
        mm += pm
        ff += pf
    if mm < ff:
        corr += 1
print corr, len(ids)
