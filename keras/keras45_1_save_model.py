from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape) # x_train = (50000, 32, 32, 3) / y_train = (50000, 1)
# print(x_test.shape, y_test.shape) # x_test = (10000, 32, 32, 3) / y_test = (10000, 1)

onehot = OneHotEncoder(sparse=False)
onehot.fit(y_train)
y_train=onehot.transform(y_train)
y_test=onehot.transform(y_test)

x_train =x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train =x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(80,(2,2),activation='relu'))
model.add(Conv2D(80,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.save('./_save/keras45_1_save_model.h5')

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=128,verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time()-start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('time',end_time)
print('loss',loss[0])
print('acc',loss[1])


