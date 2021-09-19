import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(filters=100,kernel_size=(2,2),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(80,(2,2),activation='relu'))
model.add(Conv2D(80,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss',patience=30,mode='min')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
start_time = time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=128,validation_split=0.1,verbose=2,callbacks=[es])
end_time = time.time() - start_time

#3. 평가
loss = model.evaluate(x_test,y_test)
print('time ',end_time)
print('loss ',loss[0])
print('acc' , loss[1])


# loss 0.07223863154649734
# acc 0.9824000000953674

# 결과 210721
# time 51.061420917510986
# loss 0.07350124418735504
# acc 0.9868000149726868

# 결과 210917
# time  207.64011240005493
# loss  0.05357949808239937
# acc 0.9916999936103821