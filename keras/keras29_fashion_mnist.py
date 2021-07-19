from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
 # 1. 데이터 구성
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(28,28, 1)))
model.add(Conv2D(100, (2,2), activation='relu'))
model.add(Conv2D(100, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))


# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

# epochs=1000, batch_size=32
# loss :  0.3761386573314667
# acc :  0.9003999829292297