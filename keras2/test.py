import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from sklearn.preprocessing import MinMaxScaler
import time
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 컨불루션은 4차원 데이터를 받기 떄문에 넣기전에 무조건 4차원으로 바꿔줘야함
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

onehot = OneHotEncoder(sparse=False)
onehot.fit(y_train)
y_train = onehot.fit_transform(y_train)
y_test = onehot.fit_transform(y_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(28*28,)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
es = EarlyStopping(monitor='loss', patience=10, mode='min')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,verbose=2, callbacks=[es])
end_time = time.time()-start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss', loss)

# 375/375 - 1s - loss: 0.0344 - acc: 0.9893 - val_loss: 0.1020 - val_acc: 0.9761
# Epoch 21/1000
# 375/375 - 1s - loss: 0.0323 - acc: 0.9900 - val_loss: 0.1109 - val_acc: 0.9752
# Epoch 22/1000
# 375/375 - 1s - loss: 0.0317 - acc: 0.9896 - val_loss: 0.0928 - val_acc: 0.9793
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0913 - acc: 0.9786
# time 28.841541051864624
# loss 0.09126672148704529
# acc 0.978600025177002