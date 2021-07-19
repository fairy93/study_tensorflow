from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(100, (2,2), activation='relu'))
model.add(Conv2D(100, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es], validation_split=0.2, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  3.7575008869171143
# accuracy :  0.6108999848365784