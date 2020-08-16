import numpy as np
from math import radians, cos, sin, asin, sqrt  
  
def dist(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371
    return c * r * 1000  

def getEmbeddingMat(E, V):
    return np.random.uniform(0, 1, (E, V))

def embedding(x, ev_mat):
    return np.matmul(ev_mat, x)

def vec_normalize(x):
    s = 0
    for i in x:
        s += i**2
    from math import sqrt
    s = sqrt(s)
    return [float(i)/float(s) for i in x]

def prob_normalize(x):
    s = 0
    for k, v in x:
        s += v
    return [(k, float(v)/float(s)) for k, v in x]

def build_svm(labels, features, fname):
    with open('save/'+fname, 'w') as f:
        for lab, fea in zip(labels, features):
            opt = []
            opt.append(lab)
            for i in range(len(fea)):
                opt.append(str(i) + ':' + str(fea[i]))
            f.write(' '.join(opt)+'\n')

def generate_batch(inputs, labels, batch_size):
    n = len(inputs)

    if batch_size > n:
        yield inputs, labels
        return

    for i in range(n//batch_size):
        lo = i*batch_size
        hi = i*batch_size + batch_size
        yield inputs[lo:hi], labels[lo:hi]

    if n % batch_size != 0:
        yield inputs[hi:], labels[hi:]
