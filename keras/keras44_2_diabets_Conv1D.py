import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, GlobalAveragePooling1D
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=50)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape[0], x_test.shape[0])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,
          padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2, verbose=2, callbacks=[es])
end_time = time.time()-start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)

#5. r2 예측r2
r2 = r2_score(y_test, y_predict)
print('r2', r2)

# 210728
# loss 2897.551025390625
# r2 0.4821548534785185
