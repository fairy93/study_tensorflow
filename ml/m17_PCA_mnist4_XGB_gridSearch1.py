import numpy as np
import warnings 
import time

from tensorflow.keras.datasets import mnist

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore')

parameters = [
     {"n_estimators":[100, 200, 300], 
    "learning_rate":[0.001, 0.01],
    "max_depth":[4, 5, 6], 
    "colsample_bytree":[0.6, 0.9, 1], 
    "colsample_bylevel":[0.6, 0.7, 0.9],
    "n_jobs":[-1]
    }
]

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000, 28*28)

pca = PCA(n_components=154)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = GridSearchCV(XGBClassifier(tree_method='gpu_hist'), parameters, verbose=1)

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()-start_time

#4. 평가 예측
print('수행 시간 : ', end_time)
print('최적의 파라미터 : ', model.best_estimator_)
print('score :', model.best_score_)

# 결과 dnn 210721
# time 31.354090690612793
# loss 0.0970178171992302
# acc 0.9783999919891357

# 결과 cnn 210721
# time 51.061420917510986
# loss 0.07350124418735504
# acc 0.9868000149726868

# 결과 pca = PCA(n_components=154)
