from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
datasets = load_boston()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=79)

print(np.min(x),np.max(x))

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일구현
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=0)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss',loss)

y_predict = model.predict(x_test)
# print('예측값 ',y_predict)


#5. r2 예측r2
r2= r2_score(y_test,y_predict)
print(r2)


# #결과 21.07.15
# scaler = StandardScaler()
# loss='mse', optimizer='adam'
# epochs=100, batch_size=1
# loss 20.448402404785156
# 0.7173459290703332

# #결과 21.07.15
# scaler = MinMaxScaler()
# loss='mse', optimizer='adam'
# epochs=100, batch_size=1
# loss 19.307756423950195
# 0.733112837501479

# #결과 21.07.15
# scaler = StandardScaler()
# loss='mse', optimizer='adam'
# epochs=500, batch_size=1
# loss 19.231184005737305
# r2 0.734171321628103

# #결과 21.07.15
# scaler = MinMaxScaler()
# loss='mse', optimizer='adam'
# epochs=500, batch_size=1
# loss 21.673851013183594
# r2 0.700406822357307