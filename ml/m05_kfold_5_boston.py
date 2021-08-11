from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn import datasets
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)

#2. 모델 구성
model = LinearRegression()
# acc :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 0.7128
# model = RandomForestRegressor()
# acc :  [0.92396971 0.85636426 0.81374401 0.88345256 0.89796453] 0.8751
# model = KNeighborsRegressor()
# acc :  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 0.5286
# model = DecisionTreeRegressor()
# acc :  [0.68888833 0.76795538 0.77402343 0.7430594  0.77730556] 0.7502


#3. 컴파일구현
#4. 평가, 예측
scores = cross_val_score(model, x, y, cv=Kfold)  # = r2
print('acc : ', scores, round(np.mean(scores), 4))