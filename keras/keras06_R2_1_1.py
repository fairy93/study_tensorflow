from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
# 과적합? 내신잘봐봣자 수능못보면 소요없어
# 훈련용 테스트 데이터 구분
x=np.array(range(100))
y=np.array(range(1,101))

# ,random_state=1004 고정
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True)

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

# result = model.predict(x_pred)
# print('x_pred의 예측값 : ',result)

y_predict = model.predict(x_test)
print('y_predic 의 값은 ',y_predict)

r2= r2_score(y_test,y_predict)
print(r2)