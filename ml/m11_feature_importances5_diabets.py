from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


#1. 데이터
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)
print('acc : ',acc)

print(model.feature_importances_)

# feature # 컬럼 열 