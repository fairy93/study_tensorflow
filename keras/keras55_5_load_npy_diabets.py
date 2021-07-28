import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x_data=np.load('./_save/_npy/k55_x_data_diabetes.npy')
y_data=np.load('./_save/_npy/k55_y_data_diabetes.npy')


#2. 모델 구성
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=0.8,shuffle=True, random_state=104)
model = Sequential()
model.add(Dense(5,input_dim=10))
model.add(Dense(8))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=500, batch_size=1, validation_split=0.3)

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
# epochs=120, batch_size=1
# loss :  loss 2828.759765625
# r2 : 0.527784366571096