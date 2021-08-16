import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_diabetes

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

#2. 모델
model = GradientBoostingRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
r2 = model.score(x_test,y_test)
print('r2 : ',r2)

print(model.feature_importances_)
# r2 :  0.4007861303138863
# [0.04164977 0.01363178 0.20337424 0.07613915 0.03839271 0.05267551
#  0.05290228 0.02749944 0.44313514 0.05059998]

def plot_feature_importances_dataset(modeel):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feautres")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()


