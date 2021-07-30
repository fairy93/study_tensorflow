import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

#1 데이터
x_train = np.load('./_save/_npy/k59_5_train_x.npy')
y_train = np.load('./_save/_npy/k59_5_train_y.npy')
# x_test = np.load('./_save/_npy/k59_5_test_x.npy')
# y_test = np.load('./_save/_npy/k59_5_test_y.npy')
x_pred = np.load('./_save/_npy/k59_5_pred_x.npy')
x_train, x_test,y_train,y_test = train_test_split(x_train,y_train, train_size=0.7,shuffle=True, random_state=79)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_pred.shape) # (2100, 150, 150, 3) (2100,) (900, 150, 150, 3) (900,) (1, 150, 150, 3)

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
                 verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss)
# print('acc : ',acc[-1])
# print('loss : ',loss[0])


# 내 사진으로 예측하기
y_predict = model.predict([x_pred])
print('res',(1-y_predict)*100)
# [0.6816971898078918, 0.5755555629730225]
# res [[42.678295]]
