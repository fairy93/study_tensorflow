from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBRegressor

#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model =XGBRegressor(n_estimators=100,learning_rate=0.05,n_jobs=1)

#3. 훈련
model.fit(x_train, y_train,verbose=1,eval_metric=['rmse','logloss'],eval_set=[(x_train,y_train),(x_test,y_test)])

# 저장
model.save_model('./_save/xgb_save/m23_xgb.dat')