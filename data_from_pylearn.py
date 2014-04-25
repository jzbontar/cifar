import pickle
import numpy as np

tr = pickle.load(open('/mnt/seagate/pylearn2/cifar10/pylearn2_gcn_whitened/train.pkl'))
te = pickle.load(open('/mnt/seagate/pylearn2/cifar10/pylearn2_gcn_whitened/test.pkl'))

X = np.vstack((tr.X, te.X))
y = np.concatenate((tr.y, te.y)) + 1

open('data_gcn_whitened/X', 'w').write(X.astype(np.float32).tostring())
open('data_gcn_whitened/y', 'w').write(y.astype(np.float32).tostring())
