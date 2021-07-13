from numpy.core.fromnumeric import shape, squeeze
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt

# 예제
# [1,2,3] 1x3 (3,)
# [[1,2,3]] 1x3
# [[1,2],[3,4],[5,6]] 3x2
# [[[1,2,3],[4,5,6]]] (1)*2*3
# [[1,2],[3,4],[5,6]] (1)*3*2
# [[[1],[2]],[[3],[4]]] (2)*2*1 


#1. 데이터
x=np.array([range(10)])
x=np.transpose(x)

y =np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]])
y=np.transpose(y)


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(3))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100000,batch_size=1)

#4. 평가, 예측
loss =model.evaluate(x,y)
print('loss : ',loss)

x_pred=np.array([[9]])
print(x_pred)
result=model.predict(x_pred)
print('x_pred의 예측값 : ',result)

y_predict =model.predict(x)
plt.scatter(x,)
plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()