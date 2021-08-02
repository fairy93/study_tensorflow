import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, LSTM
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28 * 1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


onehot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot.fit(y_train)
y_train = onehot.transform(y_train)
y_test = onehot.transform(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
input1 = Input(shape=(28*28, 1))
lstm = LSTM(units=50, activation='relu')(input1)
dense1 = Dense(16, activation='relu')(lstm)
output1 = Dense(10, activation="softmax")(dense1)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train, y_train, epochs=100,
          batch_size=256, verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time()-start_time


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss', loss[0])
print('acc', loss[1])
