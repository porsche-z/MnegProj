import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from xgboost import XGBClassifier
import lightgbm as lgb
import time

def pfaPmd(predction, groundTruth):
    tp = np.sum(np.logical_and(predction == 1, groundTruth == 1))
    tn = np.sum(np.logical_and(predction == 0, groundTruth == 0))
    fp = np.sum(np.logical_and(predction == 1, groundTruth == 0))
    fn = np.sum(np.logical_and(predction == 0, groundTruth == 1))
    print(str(tp)+' '+str(tn)+' '+str(fp)+' '+str(fn))
    pfa = fp/(tn+fp)
    pmd = fn/(tp+fn)
    return pfa, pmd

def DT(x_train, y_train, x_test, max_dep):
    clf = tree.DecisionTreeClassifier(max_depth=max_dep
                                      , criterion='entropy'
                                      , random_state=30
                                      , splitter="random"
                                      )
    clf = clf.fit(x_train, y_train)

    t0 = time.perf_counter()
    pred = clf.predict(x_test)
    t1 = time.perf_counter()
    timeConsume = t1-t0
    return pred, timeConsume


def SVM(x_train, y_train, x_test, kernels):
    model = svm.SVC(C=1, kernel=kernels, gamma=20, decision_function_shape='ovo')
    model.fit(x_train, y_train.ravel())

    t0 = time.perf_counter()
    pred = model.predict(x_test)
    t1 = time.perf_counter()
    timeConsume = t1-t0
    return pred, timeConsume
