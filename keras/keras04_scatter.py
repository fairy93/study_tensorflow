from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,4,3,5,7,9,3,8,12])
x_pred=[6]

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=20000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y) # 평가
print('loss : ',loss)

# result = model.predict(x_pred)
# print('x_pred의 예측값 : ',result)

y_predict =model.predict(x)
print(x_pred)
result=model.predict(x_pred)
print('x_pred의 예측값 : ',result)