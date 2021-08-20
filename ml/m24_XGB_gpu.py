import time

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

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
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, # n_jobs=8
                     tree_method='gpu_hist', 
                     predictor='gpu_predictor', # cpu_predictor
                     gpu_id=0
)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',eval_set=[(x_train, y_train), (x_test, y_test)])
print("fit time : ", time.time() - start_time)

# cpu
# fit time :  9.249370098114014
# fit time :  7.461276054382324
# fit time :  6.758653402328491
# fit time :  6.913646697998047

# gpu 
# fit time :  38.73647165298462
