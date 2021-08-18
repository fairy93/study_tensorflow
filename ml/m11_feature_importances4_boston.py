import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

#2. 모델
model = GradientBoostingRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
r2 = model.score(x_test,y_test)
print('r2 : ',r2)

print(model.feature_importances_)
# r2 :  0.8733384872125951
# [0.02256923 0.00044332 0.00103553 0.00070776 0.05204347 0.43652358
#  0.0075523  0.07097316 0.00195193 0.00884085 0.02733338 0.01139576
#  0.35862972]

def plot_feature_importances_dataset(modeel):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feautres")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()


