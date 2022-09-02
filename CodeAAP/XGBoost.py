import numpy as np
import xgboost
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE


Data = np.load('GoodDataAAP\\dataAAP.npy')
Lable = np.load('GoodDataAAP\\lableAAP.npy')

Lable = Lable - 1

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=10, train_size=0.8)

n_estimators = range(1,40,2)
scores = []
timeTrain = []
timeTest = []
for i in n_estimators:
    clf = XGBClassifier(n_estimators=i,random_state=420)
    t0 = time.time()
    clf = clf.fit(x_train, y_train)
    t1 = time.time()
    score = clf.score(x_test, y_test)
    t2 = time.time()
    scores.append(score)#the mean accuracy (your accuracy score).
    timeTrain.append(t1 - t0)
    timeTest.append(t2 - t1)

print(scores)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(n_estimators, scores, c="black", label="XGB")
plt.savefig(r"Xgboost/n_estimator.png")

learning_rates = np.arange(0.05,1.0,0.05)
scores = []
timeTrain = []
timeTest = []
for i in learning_rates:
    clf = XGBClassifier(n_estimators=30,random_state=420,learning_rate=i)
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    scores.append(score)#the mean accuracy (your accuracy score).


print(scores)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(learning_rates, scores, c="black", label="XGB")
plt.savefig(r"Xgboost/learning_Rates.png")