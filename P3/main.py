from pandas import concat
import numpy as np
import time
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import model_zoo
import data_factory
import pandas
#oat 400
#aap 300
# user_nums = [11]
dims = [1, 3, 6]
testNums = [600, 1200, 1800]
numOfModels = 6
def getAns(x_train, x_test, y_train, y_test):
    pfaCol = []
    pmdCol = []
    timeCol = []
    for depth in [12, 20, 28]:
        print('\nDT with max depth: '+str(depth))
        pred, timeCost = model_zoo.DT(x_train, y_train, x_test, depth)
        pfa, pmd = model_zoo.pfaPmd(pred, y_test)
        print('pfa: '+ str(pfa))
        print('pmd: '+str(pmd))
        pfaCol.append(pfa)
        pmdCol.append(pmd)
        timeCol.append(timeCost)
    print("____________________________________________________________")
    for kernels in ['linear', 'rbf','poly']:
        print("\nSVM with: "+kernels)

        pred, timeCost = model_zoo.SVM(x_train, y_train, x_test, kernels)
        pfa, pmd = model_zoo.pfaPmd(pred, y_test)
        print('pfa: ' + str(pfa))
        print('pmd: ' + str(pmd))
        pfaCol.append(pfa)
        pmdCol.append(pmd)
        timeCol.append(timeCost)
    return pfaCol, pmdCol, timeCol


tableDimPfa = np.full((numOfModels, 1), 0)
tableDimPmd = np.full((numOfModels, 1), 0)

for i in dims:
    x_train, x_test, y_train, y_test = data_factory.differentUserNumOrDim(2, 1200, i, 2400-480, 480)
    print("dims = " + str(i))
    pfaCol, pmdCol,timeCol = getAns(x_train, x_test, y_train, y_test)
    pfaCol = np.asarray(pfaCol)
    pfaCol = pfaCol.reshape(numOfModels, 1)
    pmdCol = np.asarray(pmdCol)
    pmdCol = pmdCol.reshape(numOfModels, 1)
    print(pfaCol)
    print(pmdCol)
    tableDimPfa = np.concatenate((tableDimPfa, pfaCol), axis=1)
    tableDimPmd = np.concatenate((tableDimPmd, pmdCol), axis=1)
    print("___________________________________________________________________________")
    print("_____________________________________________________________________________")

print(tableDimPfa)
print(tableDimPmd.shape)
print(tableDimPmd)
print(tableDimPmd.shape)
np.savetxt( "ans\\tableDimPfa.csv", tableDimPfa, delimiter="," )
np.savetxt( "ans\\tableDimPmd.csv", tableDimPmd, delimiter="," )
print("+++++++++++++++++++++++++++++dim++++++++++++++++++++++++++++++++++++++")


tableTestNumPfa = np.full((numOfModels, 1), 0)
tableTestNumPmd = np.full((numOfModels, 1), 0)
tableTestTime = np.full((numOfModels, 1), 0)
for i in testNums:
    x_train, x_test, y_train, y_test = data_factory.differentUserNumOrDim(2, 1200, 6, 600, i)
    print("test nums = " + str(i))
    pfaCol, pmdCol, timeCol = getAns(x_train, x_test, y_train, y_test)
    pfaCol = np.asarray(pfaCol)
    pfaCol = pfaCol.reshape(numOfModels, 1)
    pmdCol = np.asarray(pmdCol)
    pmdCol = pmdCol.reshape(numOfModels, 1)
    timeCol = np.asarray(timeCol)
    timeCol = timeCol.reshape(numOfModels,1)
    # print(pfaCol)
    # print(pmdCol)
    tableTestNumPfa = np.concatenate((tableTestNumPfa, pfaCol), axis=1)
    tableTestNumPmd = np.concatenate((tableTestNumPmd, pmdCol), axis=1)
    tableTestTime = np.concatenate((tableTestTime, timeCol), axis=1)

    print("___________________________________________________________________________")
    print("_____________________________________________________________________________")

# print(tableTestNumPfa)
# print(tableTestNumPfa.shape)
# print(tableTestNumPmd)
# print(tableTestNumPmd.shape)
np.savetxt( "ans\\tableTestNumPfa.csv", tableTestNumPfa, delimiter="," )
np.savetxt( "ans\\tableTestNumPmd.csv", tableTestNumPmd, delimiter="," )
np.savetxt( "ans\\tableTestTime.csv", tableTestTime, delimiter="," )
print("+++++++++++++++++++++++++++++dim++++++++++++++++++++++++++++++++++++++")



