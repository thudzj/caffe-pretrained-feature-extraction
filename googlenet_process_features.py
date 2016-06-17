# -*- coding: utf-8 -*-

import numpy as np
import hickle as hkl

if __name__ == "__main__":
    train_features = hkl.load("googlenet_train_features.hkl")
    print train_features.shape
    xs = []
    for x in train_features:
        #x = x[:,0,0]
        xs.append(np.transpose(x[0].reshape([1024, 49]), (1, 0)))
#	print np.transpose(x[0].reshape([1024, 49]), (1, 0)).shape
#	print x[0].sum(2).sum(1).shape
    xs = np.asarray(xs)
    hkl.dump(xs, "googlenet_train_features_tmp.hkl", mode="w")

    test_features = hkl.load("googlenet_test_features.hkl")
    xs = []
    for x in test_features:
        #x = x[:,0,0]
        xs.append(np.transpose(x[0].reshape([1024, 49]), (1, 0)))
    xs = np.asarray(xs)
    hkl.dump(xs, "googlenet_test_features_tmp.hkl", mode="w")

