from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris
from sklearn import datasets
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#1 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=70)
Kfold = KFold(n_splits=5, shuffle=True, random_state=66)

#2. 모델 구성
model = LinearSVC()
# acc :  [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667
# model =SVC()
# acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667
# model = KNeighborsClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
# model =LogisticRegression()
# acc :  [1.         0.96666667 1.         0.9        0.96666667] 0.9667
# model = DecisionTreeClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.93333333] 0.9533
# model = RandomForestClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96

#3. 컴파일, 훈련
#4. 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=Kfold)  # = acc
print('acc : ', scores, round(np.mean(scores), 4))
