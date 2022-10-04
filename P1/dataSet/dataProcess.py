import mat73
import numpy as np
data_dict = mat73.loadmat('AAPlantD3_5GHz_TX2b_hpol_internal_run39_pp.mat')
print(data_dict['IQdata'].shape)

train_data = []
train_label = []
test_data = []
test_label = []

labelNum = 98
simplesPerLabel = 300
trainNum = 250
testNum = 50

for i in range(labelNum):

    if i == 6:
        train_data.extend(data_dict['IQdata'].T[i * simplesPerLabel: i * simplesPerLabel + trainNum][:])
        test_data.extend(data_dict['IQdata'].T[i * simplesPerLabel + trainNum: i * simplesPerLabel + trainNum + testNum][:])
        #train_label.extend(np.full((trainNum,),i))
        #test_label.extend(np.full((testNum,),i))
        temp = np.zeros((trainNum, 4))
        temp[:, 1] = 1.0
        train_label.extend(temp)

        temp = np.zeros((testNum, 4))
        temp[:, 1] = 1.0
        test_label.extend(temp)

    elif i == 16:
        train_data.extend(data_dict['IQdata'].T[i * simplesPerLabel: i * simplesPerLabel + trainNum][:])
        test_data.extend(data_dict['IQdata'].T[i * simplesPerLabel + trainNum: i * simplesPerLabel + trainNum + testNum][:])

        temp = np.zeros((trainNum, 4))
        temp[:, 2] = 1.0
        train_label.extend(temp)

        temp = np.zeros((testNum, 4))
        temp[:, 2] = 1.0
        test_label.extend(temp)

    elif i == 26:
        train_data.extend(data_dict['IQdata'].T[i * simplesPerLabel: i * simplesPerLabel + trainNum][:])
        test_data.extend(data_dict['IQdata'].T[i * simplesPerLabel + trainNum: i * simplesPerLabel + trainNum + testNum][:])
        temp = np.zeros((trainNum, 4))
        temp[:, 3] = 1.0
        train_label.extend(temp)

        temp = np.zeros((testNum, 4))
        temp[:, 3] = 1.0
        test_label.extend(temp)

    elif i == 36:
        train_data.extend(data_dict['IQdata'].T[i * simplesPerLabel: i * simplesPerLabel + trainNum][:])
        test_data.extend(data_dict['IQdata'].T[i * simplesPerLabel + trainNum: i * simplesPerLabel + trainNum + testNum][:])
        temp = np.zeros((trainNum, 4))
        temp[:, 0] = 1.0
        train_label.extend(temp)

        temp = np.zeros((testNum, 4))
        temp[:, 0] = 1.0
        test_label.extend(temp)


train_label = np.asarray(train_label)
test_label = np.asarray(test_label)
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
print('shape of training/ testing set:', np.shape(train_data), np.shape(train_label), np.shape(test_data), np.shape(test_label))

np.save('train_data_DNN',train_data)
np.save('train_label_DNN',train_label)
np.save('test_data_DNN',test_data)
np.save('test_label_DNN',test_label)

