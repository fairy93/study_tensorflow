import time

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#1. 데이터
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

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
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

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
# time  82.42660093307495
# loss 1.6069096326828003
# acc  0.47200000286102295

# 결과 cnn 210721
# time 48.33287310600281
# loss 2.3105974197387695
# acc 0.652999997138977

# 결과 cnn 210929
# time 38.10562515258789
# loss 1.8222169876098633
# acc 0.49939998984336853