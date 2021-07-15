from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
datasets = load_boston()
x=datasets.data
y=datasets.target

# x= (x-np.min(x))/(np.max(x)-np.min(x))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_scale = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.3,shuffle=True,random_state=90)


#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(100))
model.add(Dense(16))
model.add(Dense(100))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일구현
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss',loss)

y_predict = model.predict(x_test)
print('예측값 ',y_predict)

#5. r2 예측r2
r2= r2_score(y_test,y_predict)
print(r2)

# #결과 21.07.15
# loss='mse', optimizer='adam'
# epochs=150, batch_size=1
# loss 24.682559967041016
# 0.6958809077253578

# #결과 21.07.15
# loss='mse', optimizer='adam'
# epochs=5000, batch_size=32
# loss  24.649751663208008
# 0.6962852029479341


# #결과 21.07.15
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=1
# loss 23.965864181518555
# 0.7214176909440222