import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4, ], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)

x = x.reshape(4, 3, 1)  # (batch_size, timesteps, feature)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1)
model.fit(x, y, epochs=2000, batch_size=2, verbose=1, callbacks=[es])

#4. 평가, 예측
x_input = np.array([5, 6, 7]).reshape(1, 3, 1)

res = model.predict(x_input)
print(res)

# 결과 210722
# Epoch 01515: early stopping
# [[8.]]

# param 이해하기!!
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)                120
# _________________________________________________________________
# dense (Dense)                (None, 10)                110
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 241
# Trainable params: 241
# Non-trainable params: 0
# _________________________________________________________________
