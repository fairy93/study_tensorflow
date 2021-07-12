from scipy.sparse.construct import rand
from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.model_selection import train_test_split
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=1004)


#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_split=0.3)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # 평가
print('loss : ',loss)

y_predict = model.predict([11])
print('y_predic 의 값은 ',y_predict)
