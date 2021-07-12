from scipy.sparse.construct import rand
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,shuffle=True, random_state=1004)

x_test, x_val, y_test, y_val = train_test_split(x_test,y_test,test_size=0.5,shuffle=True, random_state=1004)

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(5,input_dim=1))
# model.add(Dense(4))
# model.add(Dense(1))



# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))

# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test) # 평가
# print('loss : ',loss)

# y_predict = model.predict([11])
# print('y_predic 의 값은 ',y_predict)
