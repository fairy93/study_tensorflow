from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 출력 로스값
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred=[6]

model = Sequential()
model.add(Dense(1,input_dim=1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=5000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y) # 평가
print('loss : ',loss)

result = model.predict(x_pred)
print('x_pred의 예측값 : ',result)