from scipy.sparse.construct import rand
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=True,random_state=1004)
x_test, x_val, y_test, y_val= train_test_split(x_test,y_test,train_size=0.6,shuffle=True, random_state=1004)


print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)

#2. 모델 구성
model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # 평가
print('loss : ',loss)

y_predict = model.predict([16])
print('y_predic 의 값은 ',y_predict)

# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=100, batch_size=1
# loss :  0.00010134232434211299
# y_predic 의 값은  [[15.978352]]