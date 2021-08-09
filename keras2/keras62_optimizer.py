from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 3, 3, 5, 4, 6, 7, 8, 10,9])

#2. 모델

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일,훈련
from tensorflow.keras.optimizers import Adam,Adadelta, Adagrad,Adamax
from tensorflow.keras.optimizers import RMSprop, SGD,Nadam

optimizer = Adam(lr=0.1)
# optimizer = Adadelta(lr=0.1)
# optimizer = Adagrad(lr=0.1)
# optimizer = Adamax(lr=0.1)
# optimizer = RMSprop(lr=0.1)
# optimizer = SGD(lr=0.1)
# optimizer = Nadam(lr=0.1)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)
