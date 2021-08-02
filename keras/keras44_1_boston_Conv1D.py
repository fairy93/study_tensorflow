from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling1D, Dropout, Conv1D
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from tensorflow.python.keras.backend import conv2d

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=70)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape[0], x_test.shape[0])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,
          padding='same', input_shape=(13, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(80, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(80, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(80, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1))


#3. 컴파일 구현
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2, verbose=2, callbacks=[es])
end_time = time.time()-start_time


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#5. r2
r2 = r2_score(y_test, y_predict)

print('time', end_time)
print('loss', loss)
print('r2', r2)

# 21.07.28
# time 15.841265201568604
# loss 20.993484497070312
# r2 0.7832846084943887
