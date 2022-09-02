import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import time

Data = np.load('GoodData\\dataOat.npy')
Lable = np.load('GoodData\\lableOat.npy')

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=5, train_size=0.8)

#//////////////////////////////////////////////divide/////////////////////////////////////////////////
#train SVM


modelLinear = svm.SVC(C=1, kernel='linear', gamma=20, decision_function_shape='ovo')
t0 = time.time()
modelLinear.fit(x_train, y_train.ravel())
t1 = time.time()
print("Linear train time cost: " + str(t1-t0))
t2 = time.time()
y_pred = modelLinear.predict(x_test)
t3 = time.time()
print("Linear test time cost: " + str(t3-t2))


modelRBF = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovo')
t0 = time.time()
modelRBF.fit(x_train, y_train.ravel())
t1 = time.time()
print("rbf train time cost: " + str(t1-t0))
t2 = time.time()
y_pred = modelRBF.predict(x_test)
t3 = time.time()
print("rbf test time cost: " + str(t3-t2))

modelPoly = svm.SVC(C=1, kernel='poly', gamma=20, decision_function_shape='ovo')
t0 = time.time()
modelPoly.fit(x_train, y_train.ravel())
t1 = time.time()
print("poly train time cost: " + str(t1-t0))
t2 = time.time()
y_pred = modelPoly.predict(x_test)
t3 = time.time()
print("poly test time cost: " + str(t3-t2))

#test SVM
# resLinear = modelLinear.predict(x_test)
# resRBF = modelRBF.predict(x_test)
# resPoly = modelPoly.predict(x_test)

print("linear: " + str(modelLinear.score(x_test, y_test)))
print("rbf: " + str(modelRBF.score(x_test, y_test)))
print("poly: " + str(modelPoly.score(x_test, y_test)))

# Linear train time cost: 1.4670007228851318
# Linear test time cost: 1.355997085571289
# rbf train time cost: 2.758000612258911
# rbf test time cost: 1.921999454498291
# poly train time cost: 6.372000217437744
# poly test time cost: 0.12899994850158691
# linear: 0.8285256410256411
# rbf: 0.8935897435897436
# poly: 0.9983974358974359
