import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
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

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


onehot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot.fit(y_train)
y_train = onehot.transform(y_train)
y_test = onehot.transform(y_test)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2),
          padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(30, (2, 2), activation='relu'))
model.add(Conv2D(30, (2, 2), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
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


# loss 0.07223863154649734
# acc 0.9824000000953674

# 결과 210721
# time 51.061420917510986
# loss 0.07350124418735504
# acc 0.9868000149726868
