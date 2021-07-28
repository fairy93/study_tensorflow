from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
import numpy as np
import time

#완성하시오
# acc0.8.이상만들것

#1. 데이터
x_data=np.load('./_save/_npy/k55_x_data_wine.npy')
y_data=np.load('./_save/_npy/k55_y_data_wine.npy')

y_data = to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.7, shuffle=True, random_state=20)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구상

model = Sequential()
model.add(Dense(64, input_shape=(13,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8,
          verbose=2, validation_split=0.3, callbacks=[es])
end_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss ', loss[0])
print('acc', loss[1])

# 결과 07.21
# epochs=100, batch_size=8
# time 2.4942734241485596
# loss  0.0605735220015049
# acc 0.9814814925193787
