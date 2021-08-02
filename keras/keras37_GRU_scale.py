import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [
             8, 9, 10], [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_predict = np.array([50, 60, 70])
#-> (n,3,1)
print(x.shape, y.shape)

x = x.reshape(13, 3, 1)  # (batch_size, timesteps, feature
x_predict = x_predict.reshape(1, 3, 1)


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10,activation='relu',input_shape=(3,1)))
model.add(GRU(units=64, activation='relu', input_shape=(3, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
model.fit(x, y, epochs=200, batch_size=1,  callbacks=[es])

#4. 평가, 예측
res = model.predict(x_predict)
print(res)

# 결과 210722
# Epoch 00089: early stopping
# [[80.38959]]
