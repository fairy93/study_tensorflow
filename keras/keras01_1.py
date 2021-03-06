import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('4의 예측값 : ', result)

# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=3000, batch_size=1
# loss :  0.0
# 4의 예측값 :  [[4.]]
