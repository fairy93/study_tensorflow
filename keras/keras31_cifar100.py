from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100), kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(MaxPooling2D()) 
model.add(Conv2D(100, (3, 3), activation='relu')) 
model.add(Conv2D(100, (3, 3), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=150, callbacks=[es], validation_split=0.05, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

