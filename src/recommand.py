from nlpa_wrap import NLModel
from nlpcc_parser import SMUM_Data
from pandas import merge
from time import time
from pickle import load, dump
from nlpcc_parser import SMUM_Data
from util import prob_normalize, dist
from collections import defaultdict
from build import DataSet
from nlpa_wrap import NLModel
from numpy import NaN
import numpy as np

data = SMUM_Data()
data.load()
data.filter()
data_set = DataSet(data)
data_set.build(0.8)

checkin_m = data_set.train_male.checkin
checkin_f = data_set.train_female.checkin

test_m = data_set.test_male.checkin
ids = set(test_m.id)
for user in ids:
    chk = test_m[test_m.id == user]
    da = chk.iloc[10:,:]
    va = chk.iloc[:10,:]
    m = [9999 for i in range(len(da))]
    idx = [0 for i in range(len(da))]

    for i in range(len(checkin_m)):
        cur = checkin_m.iloc[i,:]
        for j in range(len(da)):
            a = 0
            if da.iloc[j,:].cat1 != NaN and da.iloc[j,:].cat1 == cur.cat1:
                a = 1
            if da.iloc[j,:].cat2 != NaN and da.iloc[j,:].cat2 == cur.cat2 and a == 1:
                a = 2
            if da.iloc[j,:].cat3 != NaN and da.iloc[j,:].cat3 == cur.cat3 and a == 2:
                a = 3
            dd = dist(cur.lat, cur.lng, da.iloc[j,:].lat, da.iloc[j,:].lng)*(0.8**a)
            if dd < m[j]:
                m[j] = dd
                idx[j] = i
    lst = zip(m, idx)
    lst.sort(key=lambda x:x[0])
    lst = lst[:10]
    pp = []
    for m, i in lst:
        poi = checkin_m.iloc[i,:].poi
        pp.append(poi)
    pv = []
    for i in range(10):
        poi = va.iloc[i,:].poi
        pv.append(poi)
    pp = set(pp)
    pv = set(pv)
    p = len(pp&pv)/10.0
    r = len(pp&pv)/float(len(da))
    print user, p, r
