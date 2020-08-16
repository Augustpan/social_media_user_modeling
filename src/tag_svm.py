from nlpcc_parser import SMUM_Data
from build import DataSet
from util import embedding, getEmbeddingMat, build_svm
import numpy

data = SMUM_Data()
data.load()
data.filter()
data_set = DataSet(data)
data_set.build(0.8)

tag = set(data_set.train_male.tag.tag)
tag |= set(data_set.train_fmale.tag.tag)

tag = list(tag)
dct = {tg:idx for idx, tg in enumerate(tag)}
train = []
label = []
ev_mat = getEmbeddingMat(50, len(tag))

ids = set(data_set.train_male.tag.id)
ref = data_set.train_male.tag
for i in ids:
    tg = ref[ref.id == i].tag
    seq = [0 for i in range(len(tag))]
    for t in tg:
        seq[dct[t]] = 1
    seq = embedding(numpy.array(seq), ev_mat)
    label.append('1')
    train.append(seq)
with open('save/train_male.txt', 'w') as f:
    for seq in train:
        s = ''
        for i in seq:
            s += str(i) + ' '
        s += '\n'
        f.write(s)

ids = set(data_set.train_fmale.tag.id)
ref = data_set.train_fmale.tag
for i in ids:
    tg = ref[ref.id == i].tag
    seq = [0 for i in range(len(tag))]
    for t in tg:
        seq[dct[t]] = 1
    seq = embedding(numpy.array(seq), ev_mat)
    label.append('-1')
    train.append(seq)
with open('save/train_fmale.txt', 'w') as f:
    for seq in train:
        s = ''
        for i in seq:
            s += str(i) + ' '
        s += '\n'
        f.write(s)

build_svm(label, train, 'train')

train = []
label_train = []

ids = set(data_set.test_male.tag.id)
ref = data_set.test_male.tag
for i in ids:
    tg = ref[ref.id == i].tag
    seq = [0 for i in range(len(tag))]
    for t in tg:
        seq[dct[t]] = 1
    seq = embedding(numpy.array(seq), ev_mat)
    label_train.append('1')
    train.append(seq)

test = []
label_test = []

ids = set(data_set.test_fmale.tag.id)
ref = data_set.test_fmale.tag
for i in ids:
    tg = ref[ref.id == i].tag
    seq = [0 for i in range(len(tag))]
    for t in tg:
        seq[dct[t]] = 1
    seq = embedding(numpy.array(seq), ev_mat)
    label_test.append('-1')
    test.append(seq)

# build libsvm style training set and validation set.
build_svm(label_train, train, 'svm_train.txt')
build_svm(label_test, test, 'svm_test.txt')
