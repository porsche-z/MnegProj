import torch
import mat73
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


def getG(raw_arry, dimension):
    G = np.zeros((len(raw_arry), 1))
    for i in range(0,dimension):
        G = G + np.reshape(abs(raw_arry[:, i]) ** 2, (len(raw_arry),1))#reshape as (len_of_raw_array,1)
    G = 10 * np.log10(G) + 0.9
    return G


def getFi(raw_arry, dimension):
    Fi = np.zeros((len(raw_arry), 1))
    for i in range(dimension):
        Fi = Fi + np.reshape(np.angle(raw_arry[:, i], deg=True),(len(raw_arry),1))
    Fi = Fi / dimension
    return Fi


def getPP(raw_arry, dimension):
    AbsSqare = abs(raw_arry) ** 2
    PP = np.zeros((len(raw_arry), 1))
    for i in range(len(raw_arry)):
        PP[i][0] = np.max(AbsSqare[i, :])
    return PP


def getTau(raw_arry, tau, tauAve, dimension):
    num = np.zeros((len(raw_arry), 1))
    den = np.zeros((len(raw_arry), 1))
    for i in range(dimension):
        num += np.reshape((tau[i] - tauAve) * (abs(raw_arry[:, i]) ** 2), (len(raw_arry),1))
        den += np.reshape(abs(raw_arry[:, i]) ** 2, (len(raw_arry),1))
    Tau = np.sqrt(num / den)
    return Tau


data_dict_A = mat73.loadmat('../Oats_2G_vpol_run39_pp.mat')
data_dict_B = mat73.loadmat('../Oats_2G_vpol_3115_run40_pp.mat')

print(data_dict_A['IQdata'].shape)
print(data_dict_B['IQdata'].shape)

decrease_data_A = []
decrease_data_B = []

IQdata_A = data_dict_A['IQdata'].T
IQdata_B = data_dict_B['IQdata'].T
tauA = data_dict_A['pathlossfreqrec']
tauB = data_dict_B['pathlossfreqrec']

tauAveA = np.mean(tauA)
tauAveB = np.mean(tauB)
samplesPerLabel = 400
dim = 8188

GoodDataOat = []
lableOat = []

for i in range(1, 40):
    raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
    raw_vectors_B = IQdata_B[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
    # 分别计算每个数据集的四个维度
    GA = getG(raw_vectors_A, dim)
    GB = getG(raw_vectors_B, dim)

    FiA = getFi(raw_vectors_A, dim)
    FiB = getFi(raw_vectors_B, dim)

    PPA = getPP(raw_vectors_A, dim)
    PPB = getPP(raw_vectors_B, dim)

    TauA = getTau(raw_vectors_A, tauA, tauAveA, dim)
    TauB = getTau(raw_vectors_B, tauB, tauAveB, dim)

    GoodDataOat.extend(np.concatenate((GA, FiA, PPA, TauA, GB, FiB,PPB, TauB), axis=1))
    lableOat.extend(np.full((samplesPerLabel,), i))

GoodDataOat = np.asarray(GoodDataOat)
lableOat = np.asarray(lableOat)

#noramlized_train_data = (train_data - train_data.mean())/train_data.std()

GoodDataOat = (GoodDataOat - GoodDataOat.mean())/GoodDataOat.std()
print(GoodDataOat.shape)
print(lableOat.shape)

np.save('GoodData\\dataOat', GoodDataOat)
np.save('GoodData\\lableOat', lableOat)

#split train and test

# x_train, x_test, y_train, y_test = train_test_split(GoodDataOat, lableOat, random_state=5, train_size=0.8)
# #train SVM
# model = svm.SVC(C=1, kernel='linear', gamma=20, decision_function_shape='ovo')
# model.fit(x_train, y_train.ravel())
#
# #test SVM
# result = model.predict(x_test)
# from sklearn.metrics import f1_score
# print(model.score(x_test, y_test))
