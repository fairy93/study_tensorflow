from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import datasets
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)

#2. 모델 구성
model = LinearRegression()
# acc :  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 0.4876
# model = RandomForestRegressor()
# acc :  [0.35643215 0.48904759 0.46707767 0.38701619 0.41650125] 0.4232
# model = KNeighborsRegressor()
# acc :  [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 0.3673
# model = DecisionTreeRegressor()
# acc :  [-0.19048791 -0.18784483 -0.30957217 -0.00077104 -0.02556152] -0.1428

#3. 컴파일, 훈련
#4. 평가, 예측
scores = cross_val_score(model, x, y, cv=Kfold)  # = r2
print('acc : ', scores, round(np.mean(scores), 4))