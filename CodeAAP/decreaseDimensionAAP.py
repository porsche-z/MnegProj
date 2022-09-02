import torch
import mat73
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


def getG(raw_arry, dimension):
    G = np.zeros((len(raw_arry), 1))
    for i in range(0,dimension):
        G = G + np.reshape(abs(raw_arry[:, i]) ** 2, (len(raw_arry),1))
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


data_dict_A = mat73.loadmat('../AAPlantD2 2GHz TX1 vpol internal runF_pp.mat')
data_dict_B = mat73.loadmat('../AAPlantD3_2GHz_TX2b_vpol_internal_run33_pp.mat')

print(data_dict_A['IQdata'].shape)
print(data_dict_B['IQdata'].shape)

decrease_data_A = []
decrease_data_B = []

IQdata_A = data_dict_A['IQdata'].T
IQdata_B = data_dict_B['IQdata'].T
tauA = data_dict_A['pathlossfreqrec']

tauAveA = np.mean(tauA)
samplesPerLabel = 300
dim = 8188

GoodDataOat = []
lableOat = []

def ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA):
    GA = getG(raw_vectors_A, dim)
    GB = getG(raw_vectors_B, dim)

    FiA = getFi(raw_vectors_A, dim)
    FiB = getFi(raw_vectors_B, dim)

    PPA = getPP(raw_vectors_A, dim)
    PPB = getPP(raw_vectors_B, dim)

   # TauA = getTau(raw_vectors_A, tauA, tauAveA, dim)

    GoodDataOat.extend(np.concatenate((GA, FiA, PPA, GB, FiB, PPB), axis=1))
    lableOat.extend(np.full((samplesPerLabel,), i))

for i in range(1, 70):
    if i == 6-1:
        j = 114-1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 10-1:
        j = 117-1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 14 - 1:
        j = 122 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 18 - 1:
        j = 128 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 22 - 1:
        j = 132 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 26 - 1:
        j = 136 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 32 - 1:
        j = 142 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 36 - 1:
        j = 52 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 55 - 1:
        j = 71 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 59 - 1:
        j = 74 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)
    elif i == 63 - 1:
        j = 77 - 1
        raw_vectors_A = IQdata_A[i * samplesPerLabel: i * samplesPerLabel + samplesPerLabel][:]
        raw_vectors_B = IQdata_B[j * samplesPerLabel: j * samplesPerLabel + samplesPerLabel][:]
        ProcessData(raw_vectors_A, raw_vectors_B, dim, tauA, tauAveA)


GoodDataOat = np.asarray(GoodDataOat)
lableOat = np.asarray(lableOat)
np.save('GoodDataAAP\\dataAAPNnormalize', GoodDataOat)
#noramlized_train_data = (train_data - train_data.mean())/train_data.std()

print(True in np.isnan(GoodDataOat))
print(GoodDataOat.mean())
print(GoodDataOat.std())
GoodDataOat = (GoodDataOat - GoodDataOat.mean())/GoodDataOat.std()
print(GoodDataOat.shape)
print(lableOat.shape)

np.save('GoodDataAAP\\dataAAP', GoodDataOat)
np.save('GoodDataAAP\\lableAAP', lableOat)


