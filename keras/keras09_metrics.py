from numpy.core.fromnumeric import shape, squeeze
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from matplotlib import pyplot as plt
import numpy as np
import time



#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]]) # (3,10)
x = np.transpose(x)
print(x.shape)
y =np.array([11,12,13,14,15,16,17,18,19,20])
y=np.transpose(y)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
start=time.time()
model.compile(loss='mse',optimizer='adam', metrics=['mae'])
model.fit(x,y,epochs=1000,batch_size=1,verbose=1)
end= time.time()-start
print(end)

# 4. 평가, 예측
loss =model.evaluate(x,y)
print('loss : ',loss)

x_pred=np.array([[10, 1.3, 1]])
result=model.predict(x_pred)
print('x_pred의 예측값 : ',result)
print(x.shape)

# metrics 훈련에 영향 미치지않아
# mae 뭐???
# 1. mae란 지표를 찾을것
#2. rmse란 지표를 찾을것(root) 제고하니까너무커져 그래서 루트
