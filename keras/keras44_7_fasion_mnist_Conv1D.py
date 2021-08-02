from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

onehot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot.fit(y_train)
y_train = onehot.transform(y_train)
y_test = onehot.transform(y_test)


x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,
          padding='same', input_shape=(28, 28, 1)))
model.add(Conv1D(30, 2, activation='relu'))
model.add(Conv1D(30, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


#3. 컴파일 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=2, validation_split=0.1, callbacks=[es])
end_time = time.time()-start_time

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss', loss[0])
print('loss', loss[1])


# 결과 210728
# time 48.91047215461731
# loss 0.5439285039901733
# loss 0.883899986743927
