from nlpcc_parser import SMUM_Data
from util import prob_normalize
from collections import defaultdict
from build import DataSet

data = SMUM_Data()
data.load()
data.filter()
data_set = DataSet(data)
data_set.build(0.8)

def learnProb(data_set):
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
    return dtm, dtf

def inferenceBaysin(dataset, inv = False):
    ids = set(dataset.tag.id)
    total = float(len(list(ids)))
    cc = 0
    ref = dataset.tag
    for i in ids:
        tg = list(ref[ref.id == i].tag)
        pm = 0.5
        pf = 0.5
        for t in tg:
            if t in dtm and t in dtf:
                pm *= dtm[t]
                pf *= dtf[t]
        s = pm + pf
        pm /= s
        pf /= s
        if inv:
            if pm < pf:
                cc += 1.0
        else:
            if pm > pf:
                cc += 1.0
    print "%d/%d"%(cc, total)
    return cc, total

dtm, dtf = learnProb(data_set)
cm, tm = inferenceBaysin(data_set.test_male)
cf, tf = inferenceBaysin(data_set.test_female, inv = True)
print "overall: %d/%d"%(cm+cf, tm+tf)
