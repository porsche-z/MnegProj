import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from xgboost import XGBClassifier
import lightgbm as lgb


def DT(x_train, y_train, x_test, y_test, max_dep):
    clf = tree.DecisionTreeClassifier(max_depth=max_dep
                                      , criterion='entropy'
                                      , random_state=30
                                      , splitter="random"
                                      )
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    t0 = time.time()
    clf.predict(x_test)
    t1 = time.time()
    print("DT: " + str(t1 - t0))
    return score

def KNN(x_train, y_train, x_test, y_test, num_neighbour):
    knn = KNeighborsClassifier(n_neighbors=num_neighbour)
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test, sample_weight=None)
    t0 = time.time()
    knn.predict(x_test)
    t1 = time.time()
    print("KNN: " + str(t1 - t0))
    return score

def NB(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    score = gnb.score(x_test, y_test)
    t0 = time.time()
    gnb.predict(x_test)
    t1 = time.time()
    print("NB: " + str(t1 - t0))
    return score

def SVM(x_train, y_train, x_test, y_test):
    modelRBF = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovo')
    modelRBF.fit(x_train, y_train.ravel())
    score = modelRBF.score(x_test, y_test)
    t0 = time.time()
    modelRBF.predict(x_test)
    t1 = time.time()
    print("SVM: " + str(t1 - t0))
    return score

def XGbosst(x_train, y_train, x_test, y_test, num_estimator, use_label_encoder=False):
    clf = XGBClassifier(n_estimators=num_estimator, random_state=420)
    clf = clf.fit(x_train, y_train)
    t0 = time.time()
    score = clf.score(x_test, y_test)
    t0 = time.time()
    clf.predict(x_test)
    t1 = time.time()
    print("XGboost: " + str(t1 - t0))
    return score

def Baging(x_train, y_train, x_test, y_test, num_estimator):
    clf = BaggingClassifier(n_estimators=num_estimator, random_state=420)
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    t0 = time.time()
    clf.predict(x_test)
    t1 = time.time()
    print("Baging: " + str(t1 - t0))
    return score

def GBDT(x_train, y_train, x_test, y_test, num_estimator):
    clf = GradientBoostingClassifier(n_estimators=num_estimator,random_state=420)
    clf = clf.fit(x_train, y_train)
    t0 = time.time()
    score = clf.score(x_test, y_test)
    t1 = time.time()
    print("GBDT" + str(t1 - t0))
    return score

def LGBM(x_train, y_train, x_test, y_test, num_estimator):
    clf = lgb.LGBMClassifier(n_estimators=num_estimator,random_state=420)
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    t0 = time.time()
    clf.predict(x_test)
    t1 = time.time()
    print("LGBM: " + str(t1 - t0))
    return score