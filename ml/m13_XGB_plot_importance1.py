from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


#1. 데이터
datasets = load_boston()
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
import matplotlib.pyplot as plt
import numpy as np
# feature # 컬럼 열 
def plot_feature_importances_dataset(modeel):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feautres")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

#