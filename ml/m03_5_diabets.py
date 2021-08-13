from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

#2. 모델
# model = LinearRegression()
# model.score :  0.44040657403728
# r2 :  0.44040657403728
# model = RandomForestRegressor()
# model.score :  0.42105672248752013
# r2 :  0.42105672248752013
# model = KNeighborsRegressor()
# model.score :  0.3333959730299869
# r2 :  0.3333959730299869
model = DecisionTreeRegressor()
# model.score :  0.048539587586240995
# r2 :  0.048539587586240995

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # = r2
print('model.score : ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)
