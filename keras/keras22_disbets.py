from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=67)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch_size=32,validation_split=0.1,verbose=2)

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
pred = model.predict(x_test)

#5. r2 예측
r2 = r2_score(y_test,pred)
print(r2)

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

# #결과 21.09.16
# scaler = MinMaxScaler()
# loss='mse', optimizer='adam'
# epochs=200, batch_size=32,validation_split=0.1
# r2 0.5289696254133665