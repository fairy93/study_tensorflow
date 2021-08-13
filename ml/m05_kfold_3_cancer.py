import warnings
import numpy as np

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=79)

#2. 모델
model = LinearSVC()
# acc :  [0.83333333 0.92982456 0.89473684 0.94736842 0.87610619] 0.8963
# model =SVC()
# acc :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 0.921
# model = KNeighborsClassifier()
# acc :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928
# model =LogisticRegression()
# acc :  [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 0.9385
# model = DecisionTreeClassifier()
# acc :  [0.9122807  0.92105263 0.92105263 0.87719298 0.92920354] 0.9122
# model = RandomForestClassifier()
# acc :  [0.97368421 0.96491228 0.95614035 0.95614035 0.97345133] 0.9649


#3. 컴파일 훈련
#4. 평가 예측
scores = cross_val_score(model, x, y, cv=Kfold)  # = acc
print('acc : ', scores, round(np.mean(scores), 4))
