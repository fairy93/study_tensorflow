import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28 * 1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)


onehot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot.fit(y_train)
y_train = onehot.transform(y_train)
y_test = onehot.transform(y_test)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,
          padding='same', input_shape=(28, 28)))
model.add(MaxPooling1D())
model.add(Conv1D(30, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#3. 컴파일 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000,
          batch_size=256, verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time()-start_time


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss', loss[0])
print('acc', loss[1])


# 210728
# time 32.76616168022156
# loss 0.06886795163154602
# acc 0.9819999933242798
