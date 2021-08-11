import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
from sklearn import datasets

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier') #분류 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(model,'acc : ',acc)
    except:
        print(name,'not found')
        continue
