from sklearn.ensemble import GradientBoostingClassifier


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time
from matplotlib import pyplot as plt

Data = np.load('GoodData\\dataOat.npy')
Lable = np.load('GoodData\\lableOat.npy')

Lable = Lable - 1

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=10, train_size=0.8)

n_estimators = range(1,40,2)
scores = []
timeTrain = []
timeTest = []
for i in n_estimators:
    clf = GradientBoostingClassifier(n_estimators=i,random_state=420)
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
plt.savefig(r"GDBT/n_estimator.png")
