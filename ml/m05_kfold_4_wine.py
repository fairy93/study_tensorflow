from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_wine
from sklearn import datasets
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)

#2. 모델 구성
model = LinearSVC()
# acc :  [0.86111111 0.94444444 0.52777778 0.88571429 0.88571429] 0.821
# model =SVC()
# acc :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ] 0.6457
# model = KNeighborsClassifier()
# acc :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714] 0.691
# model =LogisticRegression()
# acc :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ] 0.9608
# model = DecisionTreeClassifier()
# acc :  [0.91666667 0.97222222 0.91666667 0.82857143 0.91428571] 0.9097
# model = RandomForestClassifier()
# acc :  [1.         0.94444444 1.         0.97142857 1.        ] 0.9832

#3. 컴파일 훈련
#4. 평가, 예측
scores = cross_val_score(model, x, y, cv=Kfold)  # = acc
print('acc : ', scores, round(np.mean(scores), 4))
