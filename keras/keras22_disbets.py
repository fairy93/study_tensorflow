from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=60)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler= MaxAbsScaler()
# scaler= RobustScaler()
# scaler= QuantileTransformer()
scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
input1 = Input(shape=(10,))
dense1 = Dense(32, activation='relu', name='dense1')(input1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense5 = Dense(32, activation='relu', name='dense5')(dense2)
dense6 = Dense(16, activation='relu', name='dense6')(dense5)
dense7 = Dense(8, activation='relu', name='dense7')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32,
          validation_split=0.2, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)
# print('예측값 ',y_predict)

#5. r2 예측r2
r2 = r2_score(y_test, y_predict)
print('r2', r2)

# #결과 21.07.15
# scaler = MinMaxScaler()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.2
# loss 2891.175048828125
# r2 0.52782877892274

# #결과 21.07.15
# scaler = StandardScaler()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.2
# loss 3951.330810546875
# r2 0.3546898321085139

# #결과 21.07.15
# scaler= MaxAbsScaler()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.2
# loss 3248.349853515625
# r2 0.4694968673795943

# #결과 21.07.15
# scaler= RobustScaler()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.2
# loss 3477.415283203125
# r2 0.43208710573454434

# #결과 21.07.15
# scaler= QuantileTransformer()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.2
# loss 2889.6298828125
# r2 0.5280810556839584

# #결과 21.07.15
# scaler = PowerTransformer()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.2
# loss 3820.159423828125
# r2 0.37611193863376935
