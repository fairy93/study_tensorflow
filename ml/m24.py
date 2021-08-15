from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model =XGBRegressor(n_estimators=10000,learning_rate=0.01,tree_method='gpu_hist')

#3. 훈련
import time
start_time=time.time()
model.fit(x_train, y_train,verbose=1,eval_metric=['rmse','mae','logloss'],eval_set=[(x_train,y_train),(x_test,y_test)])

print('time : ', time.time()-start_time)

# njobs=1 / time :  11.733646869659424
# njobs=2 / time :  9.28317666053772
# njobs=4 / time :  8.664579629898071
# njobs=8 / time :  9.015531063079834
# njobs=-1 / time :  9.113933563232422

# gpu = time :  47.14815545082092