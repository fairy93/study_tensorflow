from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input, LSTM
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=70)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
input1 = Input(shape=(13,1))
lstm = LSTM(units=256, activation='relu')(input1)
dense1 = Dense(256, activation='relu')(lstm)
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(64, activation='relu')(dense4)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일구현
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
import datetime
date = datetime.datetime.now()
date_time=date.strftime("%m%d_%H%M")
filepath = './_save/ModelCheckPoint/'
filename = '.{epoch:04d}--{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k48_1",date_time,"_",filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=filepath+filename)
model.fit(x_train, y_train, epochs=500, batch_size=16,
          validation_split=0.2, verbose=2,callbacks=[es,mcp])
model.save('./_save/ModelCheckPoint/keras48_1_model_save.h5')


print("=========================1. 기본출력===========================")
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2', r2)

print("=========================2. load model ===========================")
model2 = load_model('./_save/ModelCheckPoint/keras48_1_model_save.h5')
loss = model2.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2', r2)
