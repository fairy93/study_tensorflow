from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x=np.array(range(100))
y=np.array(range(1,101))

# ,random_state=1004은 동일한 값으로 ..
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # 평가
print('loss : ',loss)

y_predict = model.predict([100])
print('y_predic 의 값은 ',y_predict)
