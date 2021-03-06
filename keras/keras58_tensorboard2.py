# cd \
# cd study
# cd _save
# cd _ graph
# dir/w
# TensorBoard --logdir=.

# 웹을 키고
# http://127.0.0.1:6006 or
# http://localhost:6006/


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,4,3,5,6,7,8,9,10])
x_pred=[6]

#2. 모델 구성
model = Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(15))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import TensorBoard
model.compile(loss='mse', optimizer='adam')
tb = TensorBoard(log_dir='./_save/_graph',histogram_freq=0,write_graph=True,write_images=True)
  
model.fit(x,y,epochs=200,batch_size=1,callbacks=[tb],validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x,y) 
print('loss : ',loss)

result = model.predict(x_pred)
print('x_pred의 예측값 : ',result)