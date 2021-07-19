from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 데이터 및 문제
# x=1,2,3,4,5
# y=1,2,3,4,5
# x=6 -> y=?

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])
x_pred = [6]

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=16)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred의 예측값은 : ', result)

# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=3500, batch_size=1
# loss :  0.380158931016922
# x_pred의 예측값은 :  [[5.721794]]

# #결과 21.07.17
# loss='mse', optimizer='adam'
# epochs=3000, batch_size=2
# loss :  0.3811221420764923
# x_pred의 예측값은 :  [[5.724585]]

# #결과 21.07.17
# loss='mse', optimizer='adam'
# epochs=2500, batch_size=2
# loss :  0.38594692945480347
# x_pred의 예측값은 :  [[5.8433414]]
