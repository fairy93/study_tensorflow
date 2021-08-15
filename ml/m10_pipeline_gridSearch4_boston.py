import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=70)

parameters = [
    {'rf__n_estimators':[100,200]},
    {'rf__max_depth' : [6,8,10,12]},
    {'rf__min_samples_leaf' : [3,5,7,10]},
    {'rf__min_samples_split' : [2,3,5,10]},
    {'rf__n_jobs' : [-1,2,4]} # cpu 몇개쓰냐
]

#2. 모델 구성
pipe = Pipeline([("scaler",MinMaxScaler()), ("rf",RandomForestRegressor())]) # 대문자 Pipeline 리스트로 넣어줘야함 파라미터
 
model = GridSearchCV(pipe, parameters,cv=Kfold)

#3. 훈련
import time
start_time = time.time()
model.fit(x_train,y_train)

#4. 평가, 예측
print("최적의 매개변수 : ",model.best_estimator_)
print("best_params_ : ",model.best_params_)
print('best_score : ', model.best_score_) # = acc(cv)

print("model.scored : ", model.score(x_test,y_test)) # =accuracy_score

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test,y_predict))

print('걸린시간 : ',time.time()-start_time)


# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('rf', RandomForestRegressor(min_samples_leaf=10))])
# best_params_ :  {'rf__min_samples_leaf': 10}
# best_score :  0.43532213728566643
# model.scored :  0.37960863745148
# r2_score :  0.37960863745148
# 걸린시간 :  16.647258281707764