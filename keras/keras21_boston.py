from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=67)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=32,validation_split=0.2,verbose=2)

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)

#5. r2 
r2 = r2_score(y_test, y_pred)
print(r2)

# #결과 21.07.15
# scaler= MaxAbsScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 13.372722625732422
# r2 0.8619536023526895

# #결과 21.07.15
# scaler = StandardScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 11.2092866897583
# r2 0.8842867254686915

# 결과 21.07.15
# scaler= MaxAbsScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 13.744830131530762
# r2 0.8581123591793408

# 결과 21.07.15
# scaler= RobustScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 13.318341255187988
# r2 0.8625149788237876

# 결과 21.07.15
# scaler= QuantileTransformer()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 19.03533363342285
# r2 0.8034985673302772

# 결과 21.07.15
# scaler = PowerTransformer()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 12.240410804748535
# r2 0.8736424397988549

# 결과 21.09.16
# scaler = MinMaxScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=32, validation_split=0.2
# r2 0.8879605568638953