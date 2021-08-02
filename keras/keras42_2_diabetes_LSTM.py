import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=30)

scaler = MinMaxScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
input1 = Input(shape=(10, 1))
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

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
model.fit(x_train, y_train, epochs=300, batch_size=16,
          validation_split=0.2, verbose=2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)
# print('예측값 ',y_predict)

#5. r2 예측r2
r2 = r2_score(y_test, y_predict)
print('r2', r2)

# 결과 210722
# loss 3365.479248046875
# r2 0.47876440210867766
