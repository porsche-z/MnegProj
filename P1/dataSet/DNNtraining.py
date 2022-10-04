import sys
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import scipy as sp

#数据处理
train_data = np.load('train_data_DNN.npy',)
train_label = np.load('train_label_DNN.npy',)
test_data = np.load('test_data_DNN.npy',)
test_label = np.load('test_label_DNN.npy',)

#noramlized_train_data = (train_data - train_data.mean())/train_data.std()
#normalized_test_data = (test_data - test_data.mean())/test_data.std()#归一化

train_data_real = train_data.real
train_data_imag = train_data.imag
train_data_ext = np.hstack((train_data_real,train_data_imag))
print(train_data_ext.shape)

test_data_real = test_data.real
test_data_imag = test_data.imag
test_data_ext = np.hstack((test_data_real,test_data_imag))
print(test_data_ext.shape)

noramlized_train_data = (train_data_ext - train_data_ext.mean())/train_data_ext.std()
normalized_test_data = (test_data_ext - test_data_ext.mean())/test_data_ext.std()#归一化

print(train_label.shape)
print(test_label.shape)



def JSDivergence(p, q):
    M = tf.multiply(0.5, tf.add(p, q))
    JSD = tf.add(K.losses.kullback_leibler_divergence(p, M), K.losses.kullback_leibler_divergence(p, M))
    JSD = tf.multiply(0.5, JSD)
    return JSD

init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.Dense(units=2048, input_dim=16376, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=960, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=240, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=120, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=40, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=4, kernel_initializer=init, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

b_size = 20
max_epochs = 50
print("Starting training ")
h = model.fit(noramlized_train_data, train_label, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")

eval = model.evaluate(normalized_test_data, test_label, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100))