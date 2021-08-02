from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)# (10000, 28, 28) (10000,)


# 컨불루션은 4차원 데이터를 받기 떄문에 넣기전에 무조건 4차원으로 바꿔줘야함
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.fit_transform(y_test).toarray()


#2. 모델 구성
model = Sequential()
model.add(Conv2D(100, kernel_size=(2, 2),
          padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(120, (2, 2), activation='relu'))
model.add(Conv2D(80, (2, 2), activation='relu'))
model.add(Conv2D(70, (2, 2), activation='relu'))
model.add(Conv2D(70, (2, 2), activation='relu'))
model.add(Conv2D(70, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))


#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2, verbose=3, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('acc', loss[1])

# loss 0.07223863154649734
# acc 0.9824000000953674
