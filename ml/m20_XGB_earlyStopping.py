from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1)

#3. 훈련(verbose 과정 보여줌, train set 명시해야 validation 지정 가능) 
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae', 'logloss'], eval_set=[
          (x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)

#4. 평가
results = model.score(x_test, y_test)
print('results : ', results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)
