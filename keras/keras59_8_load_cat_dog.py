import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

#1 데이터
x_train = np.load('./_save/_npy/k59_8_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_train_y.npy')

x_train, x_test,y_train,y_test = train_test_split(x_train,y_train, train_size=0.7,shuffle=True, random_state=79)
print(x_train.shape,y_train.shape) # (1400, 150, 150, 3) (1400,)
print(x_test.shape,y_test.shape) #(600, 150, 150, 3) (600,)

#2 모델
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
          input_shape=(150, 150, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3 컴파일 구현
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
                 verbose=2, validation_split=0.2, callbacks=[es],steps_per_epoch=32)
end_time = time.time() - start_time

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-1])
print('val_acc : ',val_acc[-1])

#acc :  0.5249999761581421
# val_acc :  0.4928571283817291