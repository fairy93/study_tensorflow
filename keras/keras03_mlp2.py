from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1.1, 1.2, 1.3, 1.4,
             1.5, 1.6, 1.5, 1.4, 1.3], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])  # (3,10)
x = np.transpose(x)
print(x.shape)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.transpose(y)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

x_pred = np.array([[10, 1.3, 1]])
result = model.predict(x_pred)
print('x_pred의 예측값 : ', result)

y_predict = model.predict(x)
plt.scatter(x[:, :1], y)
plt.plot(x, y_predict, color='red')
plt.show()
