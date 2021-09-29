import time

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#1. 데이터
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.transform(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model= Sequential()
model.add(Dense(100,input_shape=(28*28,)))
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

# loss 0.07223863154649734
# acc 0.9824000000953674

# 결과 dnn 210721
# time 31.354090690612793
# loss 0.0970178171992302
# acc 0.9783999919891357

# 결과 cnn 210721
# time 51.061420917510986
# loss 0.07350124418735504
# acc 0.9868000149726868

# 결과 dnn 210929
# time 28.614161491394043
# loss 0.18212072551250458
# acc 0.9714999794960022

# 결과 dnn 210721
# time 51.70159840583801
# loss :  0.4191642701625824
# acc :  0.881900012493133

# 결과 cnn 210721
# time 52.32506084442139
# loss 0.4858587980270386
# loss 0.9111999869346619

# 결과 dnn 210929
# time 21.635500192642212
# loss 0.5280010104179382
# acc 0.8835999965667725