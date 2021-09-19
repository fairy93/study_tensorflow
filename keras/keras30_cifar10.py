import time

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(filters=100,kernel_size=(2,2),padding='same',input_shape=(32,32,3)))
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))


#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss',patience=30,mode='min')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
start_time = time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=256,validation_split=0.1,verbose=2,callbacks=[es])
end_time = time.time() - start_time

#3. 평가
loss = model.evaluate(x_test,y_test)
print('time ',end_time)
print('loss ',loss[0])
print('acc' , loss[1])


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


# 결과 210917 StandardScaler()
# time 40.282625913619995
# loss 2.5071310997009277
# acc 0.6467999815940857