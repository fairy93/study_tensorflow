from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7])
y_train=np.array([1,2,3,4,5,6,7])

# 과적합? 내신잘봐봣자 수능못보면 소요없어
# 훈련용 테스트 데이터 구분
x_test=np.array([8,9,10])
y_test=np.array([8,9,10])
x_pred=[6]

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=10000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # 평가
print('loss : ',loss)

# result = model.predict(x_pred)
# print('x_pred의 예측값 : ',result)

y_predict = model.predict([11])
print('y_predic 의 값은 ',y_predict)