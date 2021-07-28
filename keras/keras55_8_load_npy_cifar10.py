from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# 1. 데이터

# cifar10 npy save

x_train = np.load('./_save/_npy/k55_x_train_cifar10.npy')
y_train =np.load('./_save/_npy/k55_y_train_cifar10.npy')
x_test = np.load('./_save/_npy/k55_x_test_cifar10.npy')
y_test = np.load('./_save/_npy/k55_y_test_cifar10.npy')

onehot = OneHotEncoder(sparse=False)
onehot.fit(y_train)
y_train=onehot.transform(y_train)
y_test=onehot.transform(y_test)

x_train =x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train =x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(30,(2,2),activation='relu'))
model.add(Conv2D(30,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=128,verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time()-start_time

model.save('./_save/keras30_1_save_model_2.h5')
#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('time',end_time)
print('loss',loss[0])
print('acc',loss[1])


# loss :  3.7575008869171143
# acc :  0.6108999848365784

# 결과 210721 MinMaxScaler()
# time 48.33287310600281
# loss 2.3105974197387695
# acc 0.652999997138977

# 결과 210721 StandardScaler()
# time 40.282625913619995
# loss 2.5071310997009277
# acc 0.6467999815940857