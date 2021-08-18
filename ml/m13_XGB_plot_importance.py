# feature_importance를 돌려 데이터가 20~25%미만인 데이터를 지우고 데이터를 재구성 한뒤
# 모델별로 결과 구하기

import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from xgboost import plot_importance

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
r2 = model.score(x_test, y_test)
print('r2 : ', r2)

print(model.feature_importances_)


# def plot_feature_importance_dataset(model):
#       n_features = datasets.data.shape[1]
#       plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#       plt.yticks(np.arange(n_features), datasets.feature_names)
#       plt.xlabel("Feature Importances")
#       plt.ylabel("Features")
#       plt.ylim(-1, n_features)

# plot_feature_importance_dataset(model)
# plt.show()

plot_importance(model)
plt.show()
