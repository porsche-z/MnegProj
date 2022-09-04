import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from xgboost import XGBClassifier


def DT(x_train, y_train, x_test, y_test, max_dep):
    clf = tree.DecisionTreeClassifier(max_depth=max_dep
                                      , criterion='entropy'
                                      , random_state=30
                                      , splitter="random"
                                      )
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    return score

def KNN(x_train, y_train, x_test, y_test, num_neighbour):
    knn = KNeighborsClassifier(n_neighbors=num_neighbour)
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test, sample_weight=None)
    return score

def NB(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    score = gnb.score(x_test, y_test)
    return score

def SVM(x_train, y_train, x_test, y_test):
    modelRBF = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovo')
    modelRBF.fit(x_train, y_train.ravel())
    score = modelRBF.score(x_test, y_test)
    return score

def XGbosst(x_train, y_train, x_test, y_test, num_estimator, use_label_encoder=False):
    clf = XGBClassifier(n_estimators=num_estimator, random_state=420)
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    return score
