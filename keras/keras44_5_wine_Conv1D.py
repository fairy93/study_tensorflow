from sklearn import datasets
from sklearn.datasets import load_iris
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import time
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_wine
#완성하시오
# acc0.8.이상만들것

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=20)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,
          padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(32, 1, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8,
          verbose=2, validation_split=0.3, callbacks=[es])
end_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss ', loss[0])
print('acc', loss[1])

# 210728
time 9.119587182998657
# loss  0.24941852688789368
# acc 0.9814814925193787