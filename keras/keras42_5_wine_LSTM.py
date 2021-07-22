from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
import numpy as np
import time

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

#2. 모델 구상

input1 = Input(shape=(13, 1))
lstm = LSTM(units=256, activation='relu')(input1)
dense1 = Dense(256, activation='relu')(lstm)
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(64, activation='relu')(dense4)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
output1 = Dense(3,activation="softmax")(dense7)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=4,
          verbose=2, validation_split=0.2, callbacks=[es])
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

# 결과 07.22
# time 26.595943927764893
# loss  0.4859378933906555
# acc 0.8333333134651184