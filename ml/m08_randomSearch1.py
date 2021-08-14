import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=70)

parameters = [
    {'n_estimators':[100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1,2,4]} # cpu 몇개쓰냐
]

#2. 모델
model = RandomizedSearchCV(RandomForestClassifier(), parameters,cv=Kfold,verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
print("최적의 파라미터 : ",model.best_estimator_)
print('best_score : ', model.best_score_) # = acc(cv)

print("model.scored : ", model.score(x_test,y_test)) # =accuracy_score

y_pred = model.predict(x_test)
print("accuracy_score : ",accuracy_score(y_test,y_pred))

# 최적의 파라미터 :  RandomForestClassifier(n_jobs=4)
# best_score :  0.9583333333333334
# model.scored :  1.0
# accuracy_score :  1.0