from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from sklearn.preprocessing import MinMaxScaler
import time
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

onehot = OneHotEncoder(sparse=False)
onehot.fit(y_train)
y_train = onehot.fit_transform(y_train)
y_test = onehot.fit_transform(y_test)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(32*32*3,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(164, activation='relu'))
model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2, callbacks=[es], verbose=2)
end_time = time.time()-start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('time ', end_time)
print('loss', loss[0])
print('acc ', loss[1])

# 결과 dnn 210721
# time  82.42660093307495
# loss 1.6069096326828003
# acc  0.47200000286102295

# 결과 cnn 210721
# time 48.33287310600281
# loss 2.3105974197387695
# acc 0.652999997138977
