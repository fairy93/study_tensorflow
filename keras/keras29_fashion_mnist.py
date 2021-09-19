import time

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

#1. 데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

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
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(Conv2D(50,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
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


# 결과 210721
# time 52.32506084442139
# loss 0.4858587980270386
# loss 0.9111999869346619

#  결과 210917
# time  106.20374059677124
# loss  0.801498532295227
# acc 0.9122999906539917