# Timeseries data make train, test example function
# preprocess, traintest split, early stop

import time
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
import numpy as np

x_data = np.array(range(1, 101))
x_pred = np.array(range(96, 106))

size1 = 6
size2 = 5


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i: (i + size)]
        aaa.append(subset)
    return np.array(aaa)


dataset = split_x(x_data, size1)

x_pred = split_x(x_pred, size2)  # (6, 5)

x = dataset[:, :-1]  # (95, 5)
y = dataset[:, -1]  # (95,)

print(x.shape, y.shape, x_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, shuffle=False, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*1)
print(x_train.shape)
print(x_test.shape)
print(x_pred.shape)

# 2. model

model = Sequential()
# model.add(LSTM(units=10, activation='relu', input_shape=(5, 1)))
model.add(Dense(64, activation='relu', input_dim=5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. compile train

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=64,
          validation_split=0.1, callbacks=[es])
end_time = time.time() - start_time

# 4. pred eval
y_pred = model.predict(x_test)
print("time : ", end_time)
print('y_pred : \n', y_pred)

print(y_pred.shape)

y_pred = y_pred.reshape(y_pred.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

print(y_pred.shape)  # (6,)
print(y_test.shape)  # (6,)


def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


rmse = RMSE(y_test, y_pred)
print('rmse score : ', rmse)

r2 = r2_score(y_test, y_pred)
print('R^2 score : ', r2)

res = model.predict(x_pred)
print('predict :', res)
