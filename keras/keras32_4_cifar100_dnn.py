import time

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#1. 데이터
(x_train, y_train), (x_test,y_test) = cifar100.load_data()

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)
y_train = y_train.reshape(50000, 1)
y_test = y_test.reshape(10000, 1)

onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.transform(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model= Sequential()
model.add(Dense(100,input_shape=(32*32*3,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss',patience=30,mode='min')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
start_time = time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=128,validation_split=0.1,callbacks=[es], verbose=2)
end_time = time.time()-start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('time',end_time)
print('loss',loss[0])
print('acc',loss[1])


# 결과 dnn 210721
# time 24.471125841140747
# loss 3.7037198543548584
# acc 0.21150000393390656

# 결과 cnn 210721
# time 74.7539279460907
# loss 5.903629779815674
# acc 0.3456000089645386

# 결과 dnn 210929
# time 42.30315089225769
# loss 5.1053948402404785
# acc 0.21150000393390656