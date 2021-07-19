# overfit을 극복하자
#1. 전체훈련 데이터 마니마니
#2. 노말라이제이션
#3. dropout

# fully connected layer


# example cifar10
# make perfect model
# top 3 coffee

from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

# 수정1
# x_train = x_train.reshape(50000, 32, 32, 3)/255. # (50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)/255. # (10000, 32, 32, 3)

# 수정1
x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3)# (10000, 32, 32, 3)

# # 수정1
# x_train = x_train/255.
# x_test = x_test/255.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# 2 스탠다드스케일러해야해
# x_train = scaler.fit(x_train)
# x_train=scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 수정1
x_train = x_train.reshape(50000, 32,32,3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32,32,3) # (10000, 32, 32, 3)
# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (50000, 100)
# y_test = one.transform(y_test).toarray() # (10000, 100)

# to_categorical 원핫과같음 2차원 -1은 전체
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,Dropout,GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())        

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())                   

model.add(Conv2D(64, (2, 2), activation='relu'))   
model.add(Dropout(0.2)) 
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPool2D())  

# model.add(Flatten())                                              
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
import time
start_time = time.time()
hist=model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time()-start_time
# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print("검사시간 : ",end_time)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[1])

#. 1
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2.
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])
plt.show()
