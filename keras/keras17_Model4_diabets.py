import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes

#1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=104)

#2. 모델 구성
input1 = Input(shape=(10,))
dense1 = Dense(16)(input1)
dense2 = Dense(32)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(128)(dense3)
dense5 = Dense(64)(dense4)
dense6 = Dense(32)(dense5)
dense7 = Dense(16)(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss',loss)
y_predict = model.predict(x_test)
print('예측값 ',y_predict)

#5. r2 예측r2
r2= r2_score(y_test, y_predict)
print(r2)

# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=100, batch_size=1
# loss 2806.576904296875
# r2 : 0.5314874525355195