from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7])   # 훈련, 공유하는 거
y_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])   # 평가하는거!!
y_test=np.array([8,9,10])
x_val = np.array([11,12,13])   # 검증
y_val = np.array([11,12,13])

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # 평가
print('loss : ',loss)

y_predict = model.predict([11])
print('y_predic 의 값은 ',y_predict)
