from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델 구성
model = LinearRegression()
# model.score :  0.7392832939303402
# r2 :  0.7392832939303402
# model = RandomForestRegressor()
# model.score :  0.8551509503623276
# r2 :  0.8551509503623276

# #3. 컴파일구현
model.fit(x_train, y_train)

# #4. 평가, 예측
result = model.score(x_test, y_test) # = r2
print('model.score : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
