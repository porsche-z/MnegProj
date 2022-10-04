import numpy as np
from sklearn.model_selection import train_test_split
#OAT为400
rawData = np.load('GoodDataAAP\\dataAAP.npy')
rawLable = np.load('GoodDataAAP\\lableAAP.npy')

def differentUserNumOrDim(userNum, samplesPerLabel, dim, trainSize, testSize):
    if(trainSize+testSize > samplesPerLabel):
        print("Too much samples, only "+ str(samplesPerLabel))
    data = []
    label = []
    j = 0
    for i in range(0, userNum):
        tmpData = rawData[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel, 0:dim]
        tmpLabel = np.full((samplesPerLabel,), i)
        data.extend(tmpData)
        label.extend(tmpLabel)
    data = np.asarray(data)
    label = np.asarray(label)
    x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=5, train_size=trainSize, test_size=testSize)
    return x_train, x_test, y_train, y_test

