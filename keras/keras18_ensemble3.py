from operator import concat
import numpy as np
from tensorflow.python.keras.layers.core import Activation
from sklearn.model_selection import train_test_split

x1 = np. array([range(100),range(301,401),range(1,101)])
x1 = np.transpose(x1)
y1 = np.array(range(1001,1101))
y2 = np.array(range(1901,2001))


x1_train, x1_test, y1_train, y1_test, y2_train, y2_test= train_test_split(x1,y1,y2,train_size=0.7,shuffle=True, random_state=1004)


#2. 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1 # input = 1 , output=2
input1 = Input(shape=(3,))
dense1= Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(3, activation='relu', name='dense2')(dense1)
dense3 = Dense(2, activation='relu', name='dense3')(dense2)
output1 = Dense(3, name='output1')(dense3)

last_output1 = Dense(1)(output1)
last_output2 = Dense(1)(output1)

model = Model(inputs=input1,outputs=[last_output1,last_output2])
model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train],[y1_train,y2_train], epochs=100,batch_size=9,verbose=1)

#4. 평가, 예측
results=model.evalute([x1_test],[y1_test,y2_test])
print('loss: ',results[0])
print('metrics[mae] :', results[1])
