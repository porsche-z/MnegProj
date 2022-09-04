import time
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

Data = np.load('GoodDataAAP\\dataAAP.npy')
Lable = np.load('GoodDataAAP\\lableAAP.npy')

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=5, train_size=0.8)

#train KNN
print("KNN")

x = np.arange(1,31)
y_score = []
y_time = []
for i in range(1,31):

    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(x_train, y_train)
    y_score.append(knn.score(x_test,y_test,sample_weight=None))
    time_start = time.time()  # 记录开始时间
    y_predict = knn.predict(x_test)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start
    y_time.append(time_sum)

plt.title('KNN')
plt.xlabel('n_neighbors')
plt.ylabel('score')
plt.plot(x, y_score)
plt.show()
plt.title('KNN')
plt.xlabel('n_neighbors')
plt.ylabel('time')
plt.plot(x, y_score)
plt.savefig(r"KNNres\timeCost.png")