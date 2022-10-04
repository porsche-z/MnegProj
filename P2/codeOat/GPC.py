import numpy as np
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

Data = np.load('GoodData\\dataOat.npy')
Lable = np.load('GoodData\\lableOat.npy')

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=5, train_size=0.8)

print("data loaded")
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,
        random_state=0).fit(x_train, y_train)
print(gpc.score(x_test, y_test))
