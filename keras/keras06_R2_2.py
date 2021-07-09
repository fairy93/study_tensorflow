from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import optimizers
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.python.ops.gen_dataset_ops import optimize_dataset

# 출력 로스값
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred=[6]

model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(5))
model.add(Dense(12))
model.add(Dense(1))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='sgd')

model.fit(x,y,epochs=15000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y) # 평가
print('loss : ',loss)

y_predict = model.predict(x)
print('x_pred의 예측값 : ',y_predict)

r2= r2_score(y,y_predict)
print(r2)

# 과적합 = 오버피팅