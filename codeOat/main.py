from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import model_zoo
import data_factory

#oat 400
#aap 300
user_nums = [10, 20, 30, 39]
dims = [1, 4, 8]
testNums = [10,20,30,40]
trainNums = [50, 150, 250, 300]

def getAns(x_train, x_test, y_train, y_test):
    print("DT SCORE: ")
    print(model_zoo.DT(x_train, y_train, x_test, y_test, max_dep = 12))
    print("KNN SCORE: ")
    print(model_zoo.KNN(x_train, y_train, x_test, y_test, num_neighbour=10))
    print("NB SCORE: ")
    print(model_zoo.NB(x_train, y_train, x_test, y_test))
    print("SVM SCORE: ")
    print(model_zoo.SVM(x_train, y_train, x_test, y_test))
    print("XGBoost SCORE: ")
    print(model_zoo.XGbosst(x_train, y_train, x_test, y_test, num_estimator=30))


for i in user_nums:
    x_train, x_test, y_train, y_test = data_factory.differentUserNumOrDim(i, 400, 8, 0.8, 0.2)
    print("user_nums = " + str(i))
    getAns(x_train, x_test, y_train, y_test)
    print("-----------------------------------------------------------------------------------------")
print("++++++++++++++++++++++++++++++++++++++++user_num_finished+++++++++++++++++++++++++++++++++++++++++++++++++")
for i in dims:
    x_train, x_test, y_train, y_test = data_factory.differentUserNumOrDim(30, 400, i, 0.8, 0.2)
    print("dims = " + str(i))
    getAns(x_train, x_test, y_train, y_test)
    print("-----------------------------------------------------------------------------------------")
print("++++++++++++++++++++++++++++++++++++++++dims_finished+++++++++++++++++++++++++++++++++++++++++++++++++")

for i in trainNums:
    x_train, x_test, y_train, y_test = data_factory.differentUserNumOrDim(30, 400, 8, i*30, 80*30)
    print("trainNum = "+str(i))
    getAns(x_train, x_test, y_train, y_test)
    print("-----------------------------------------------------------------------------------------")
print("++++++++++++++++++++++++++++++++++++++++train_num_finished+++++++++++++++++++++++++++++++++++++++++++++++++")



