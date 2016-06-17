# -*- coding: utf-8 -*-

import numpy as np
import hickle as hkl
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score

def run(model_name):
    # 訓練データ読み込み
    print "==> loading train data from %s" % (model_name + "_train_(features|labels).hkl")
    train_features = hkl.load(model_name + "_train_features_tmp.hkl")
    train_labels = hkl.load(model_name + "_train_labels.hkl")
    print "train_features.shape =", train_features.shape
    print "train_labels.shape =", train_labels.shape

    svm = LinearSVC(C=1.0)
    
    # print "==> training and test"
    # X_train = train_features[-1000:]
    # T_train = train_labels[-1000:]
    # X_test = train_features[:-1000]
    # T_test = train_labels[:-1000]
    # svm.fit(X_train, T_train)
    # Y_test = svm.predict(X_test)
    # print confusion_matrix(T_test, Y_test)
    # print accuracy_score(T_test, Y_test)
    # print classification_report(T_test, Y_test)
    
    # 10分割交差検定
#    print "==> cross validation"
#    scores = cross_val_score(svm, train_features, train_labels, cv=10)
#    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())

    # 全訓練データで学習
    svm.fit(train_features, train_labels)
    
    # テストデータ読み込み
    print "==> loading test data from %s" % (model_name + "_test_(features|labels).hkl")
    test_features = hkl.load(model_name + "_test_features_tmp.hkl")
    test_labels = hkl.load(model_name + "_test_labels.hkl")
    print test_labels.shape
    
    # 予測結果をCSVに書き込む
    print "==> predicting and writing"
    predicted_labels = svm.predict(test_features)
    print predicted_labels.shape
    predicted_labels = predicted_labels.reshape(test_labels.shape)
    print sum(predicted_labels == test_labels) * 1.0 / len(test_features)
    f = open("temp", "w")
    for i, j in zip(predicted_labels, test_labels):
        f.write("%d, %d\n" %(i,j))
    f.close()
        
if __name__ == "__main__":
    #run("alexnet")
    #run("vgg16_fc7")
    #run("vgg16_fc6")
    run("googlenet")
