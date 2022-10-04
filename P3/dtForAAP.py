import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
import time

from P3 import model_zoo

Data = np.load('goodDataP3\\dataAAP.npy')
Lable = np.load('goodDataP3\\lableAAP.npy')

x_train, x_test, y_train, y_test = train_test_split(Data, Lable, random_state=5, train_size=0.8)

#//////////////////////////////////////////////divide/////////////////////////////////////////////////"entropy"
#train DT
#entropy
def train_And_test(criterion):
    PFAs = []
    PMDs = []
    for i in range(6, 20):
        clf = tree.DecisionTreeClassifier(max_depth=i + 1
                                          , criterion=criterion
                                          , random_state=30
                                          , splitter="random"
                                          )
        t0 = time.time()
        clf = clf.fit(x_train, y_train)
        # t1 = time.time()
        # score = clf.score(x_test, y_test)
        # t2 = time.time()
        y_pred = clf.predict(x_test)
        print(y_pred)
        t3 = time.time()
        pfa, pmd = model_zoo.pfaPmd(y_pred, y_test)
        PMDs.append(pmd)#the mean accuracy (your accuracy score).
        PFAs.append(pfa)
        # timeTest.append(t3 - t2)

    return PFAs, PMDs

def draw(x, y, z, fileName, x_lable, y_lable, z_lable):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-', label=y_lable)
    print(y)
    print(z)

    ax2 = ax.twinx()
    ax2.plot(x, z, '-r', label=z_lable)
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_lable)
    ax2.set_ylabel(z_lable)

    plt.savefig(fileName)


PFA, PMD = train_And_test("gini")
print(len(PFA))
print(len(PMD))
#
x = np.arange(6, 20)
print(x)
dirName = "F:\\studing_project\\MnegRui\\P3"
draw(x,PFA,PMD,dirName+"score.png", "maxDepth", "PFA", "PMD")
# draw(x,timeTrainE,timeTrainG,dirName+"trainTimeCost.png", "maxDepth", "entropy", "gini")
# draw(x,timeTrainE,timeTrainG,dirName+"testTimeCost.png", "maxDepth", "entropy", "gini")





