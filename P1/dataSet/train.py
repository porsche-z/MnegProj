from numpy.core.fromnumeric import shape
from sklearn.datasets import load_digits
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import tensorflow as tf

train_data = np.load('train_data.npy',)
train_label = np.load('train_label.npy',)
test_data = np.load('test_data.npy',)
test_label = np.load('test_label.npy',)

noramlized_train_data = (train_data - train_data.mean())/train_data.std()
normalized_test_data = (test_data - test_data.mean())/test_data.std()#归一化

train_data_real = train_data.real
train_data_imag = train_data.imag
train_data_ext = np.hstack((train_data_real,train_data_imag))
print(train_data_ext.shape)

test_data_real = test_data.real
test_data_imag = test_data.imag
test_data_ext = np.hstack((test_data_real,test_data_imag))
print(test_data_ext.shape)

model = OneVsRestClassifier(SVC(kernel='rbf',probability=True))
print("[INFO] Successfully initialize a new model !")


version = tf.__version__
gpu_ok = tf.test.is_gpu_available()
print("tf version:",version,"\nuse GPU",gpu_ok)
print(tf.config.list_physical_devices('GPU'))

print("[INFO] Training the model…… ")

clt = model.fit(train_data_ext,train_label)
print("[INFO] Model training completed !")

testPred = clt.predict(test_data_ext)
overallAcc = metrics.accuracy_score(testPred,test_label)
print("overall accuracy: %f"%(overallAcc))