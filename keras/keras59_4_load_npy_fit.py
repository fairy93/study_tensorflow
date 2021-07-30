import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time

#1 데이터
x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')



print(x_train.shape, y_train.shape) # (5, 150, 150, 3) (5,) 
print(x_test.shape, y_test.shape) # (5, 150, 150, 3) (5,)

#2 모델 구성
model = Sequential()
model.add(Conv2D(100, kernel_size=(2, 2), padding='same', input_shape=(150,150,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))   
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), padding='same', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), padding='same', activation='relu')) 
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))


#3 컴파일 구현
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, validation_split=0.2, verbose=2, callbacks=[es], shuffle=True, batch_size=64,steps_per_epoch=32)
end_time = time.time() - start_time

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-1])
print('loss : ',loss[0])

# acc :  1.0
# loss :  0.0003209338756278157