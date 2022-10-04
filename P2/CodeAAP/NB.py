import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

Data = np.load('GoodDataAAP\\dataAAP.npy')
Lable = np.load('GoodDataAAP\\lableAAP.npy')

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=5, train_size=0.8)

gnb=GaussianNB()
#train Time
t0 = time.time()
gnb.fit(x_train, y_train)
t1 = time.time()
#test time
t2 = time.time()
y_pred = gnb.predict(x_test)
t3 = time.time()
print("GaussianKernel" + str(gnb.score(x_test, y_test)))
print("train time cost: " + str(t1-t0))
print("test time cost: " + str(t3-t2))

# GaussianKernel0.9980769230769231
# train time cost: 0.003999948501586914
# test time cost: 0.005002260208129883