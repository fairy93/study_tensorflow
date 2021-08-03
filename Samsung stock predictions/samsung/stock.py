from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, Dropout, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import datetime

#1. 데이터
x1 = pd.read_csv(
    './Samsung stock predictions/_data/삼성전자 주가 20210721.csv', encoding='EUC_KR')
x2 = pd.read_csv(
    './Samsung stock predictions/_data/SK주가 20210721.csv', encoding='EUC_KR')

#1-1 x1=samsung x2=sk
x1 = x1[['시가', '고가', '저가', '거래량', '종가']]
x1 = x1[:2601]
x1 = x1[::-1]
y = x1['종가']
y = y[:-19]
x1 = x1[['시가', '고가', '저가', '거래량']]

x2 = x2[['시가', '고가', '저가', '거래량', '종가']]
x2 = x2[:2601]
x2 = x2[::-1]

x1 = x1.to_numpy()
x2 = x2.to_numpy()
y = y.to_numpy()

size = 20
def split_x(dataset, size):
    temp = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i: (i + size), :]
        temp.append(subset)
    return np.array(temp)


x1 = split_x(x1, size)
x2 = split_x(x2, size)
x1_pred = x1[-1, :]
x2_pred = x2[-1, :]
x1 = x1.reshape(x1.shape[0], x1.shape[1]*x1.shape[2])
x2 = x2.reshape(x2.shape[0], x2.shape[1]*x2.shape[2])
x1_pred = x1_pred.reshape(1, 80)
x2_pred = x2_pred.reshape(1, 100)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=True)

scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_pred = scaler.transform(x1_pred)

x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_pred = scaler.transform(x2_pred)

x1_train = x1_train.reshape(x1_train.shape[0], 20, 4)
x1_test = x1_test.reshape(x1_test.shape[0], 20, 4)
x2_train = x2_train.reshape(x2_train.shape[0], 20, 5)
x2_test = x2_test.reshape(x2_test.shape[0], 20, 5)
x1_pred = x1_pred.reshape(x1_pred.shape[0], 20, 4)
x2_pred = x2_pred.reshape(x2_pred.shape[0], 20, 5)

# 2 모델 구성
#2-1 모델1
input1 = Input(shape=(20, 4))
x1_1 = Conv1D(filters=100, kernel_size=3, activation='relu')(input1)
x1_2 = Conv1D(filters=100, kernel_size=3, activation='relu')(x1_1)
x1_3 = Dropout(0.2)(x1_2)
x1_4 = Conv1D(filters=80, kernel_size=3, activation='relu')(x1_3)
x1_5 = Conv1D(filters=64, kernel_size=3, activation='relu')(x1_4)
x1_6 = Dropout(0.2)(x1_5)
x1_7 = Conv1D(filters=64, kernel_size=3, activation='relu')(x1_6)
x1_8 = Conv1D(filters=32, kernel_size=3, activation='relu')(x1_7)
x1_9 = Flatten()(x1_8)
output1 = Dense(16, activation='relu')(x1_9)

#2-2 모델2
input2 = Input(shape=(20, 5))
x2_1 = Conv1D(filters=100, kernel_size=3, activation='relu')(input1)
x2_2 = Conv1D(filters=100, kernel_size=3, activation='relu')(x2_1)
x2_3 = Dropout(0.2)(x2_2)
x2_4 = Conv1D(filters=80, kernel_size=3, activation='relu')(x2_3)
x2_5 = Conv1D(filters=64, kernel_size=3, activation='relu')(x2_4)
x2_6 = Dropout(0.2)(x2_5)
x2_7 = Conv1D(filters=64, kernel_size=3, activation='relu')(x2_6)
x2_8 = Conv1D(filters=32, kernel_size=3, activation='relu')(x2_7)
x2_9 = Flatten()(x2_8)
output2 = Dense(16, activation='relu')(x2_9)


merge = concatenate([output1, output2])
merge1 = Dense(16, activation='relu')(merge)
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)
model = Model(inputs=[input1, input2], outputs=last_output)

#3 컴파일 구현
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath = 'Samsung stock predictions/_save/ModelCheckPoint/'
filename = '.{epoch:04d}-{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "samsung_", date_time, "_", filename])
cp = ModelCheckpoint(monitor='val_loss', mode='auto', patience=10,
                     verbose=1, save_best_only=True, filepath=modelpath)

es = EarlyStopping(monitor='val_loss', patience=50, verbose=1,
                   mode='auto', restore_best_weights=True)

model.fit([x1_train, x2_train], y_train, epochs=5000, batch_size=128,
          verbose=2, validation_split=0.1, callbacks=[es, cp])

#4 평가 예측
loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_pred, x2_pred])

print(y_predict)

# 7.23 종가 예측 [[79613.93]]
