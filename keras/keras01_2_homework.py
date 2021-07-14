from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 데이터 및 문제
# x=1,2,3,4,5
# y=1,2,3,4,5
# x=6 -> y=?

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred=[6]

#2. 모델 구성
model= Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=3500, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred의 예측값은 : ',result)

# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=3500, batch_size=1
# loss :  0.380158931016922
# x_pred의 예측값은 :  [[5.721794]]