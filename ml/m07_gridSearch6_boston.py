import warnings

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=70)

# 'n_estimator' 몇번훈련시킬것이냐
parameters = [
    {'n_estimators':[100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1,2,4]} # cpu 몇개쓰냐
]

#2. 모델
model = GridSearchCV(RandomForestRegressor(), parameters,cv=Kfold)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
print("최적의 파라미터 : ",model.best_estimator_)
print('best_score : ', model.best_score_) # =r2(cv)

print('model.score : ', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('r2_score : ', r2_score(y_test,y_pred))

# 최적의 파라미터 :  RandomForestRegressor(max_depth=12)
# best_score :  0.8527585304580502
# model.score :  0.8875871976782805
# r2_score :  0.8875871976782805