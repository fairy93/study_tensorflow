from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
 # 1. 데이터 구성
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler= MaxAbsScaler()
# scaler= RobustScaler()
# scaler= QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Dense(100,input_shape=(28*28,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

# epochs=1000, batch_size=32
# loss :  0.3677433729171753
# acc :  0.8726000189781189