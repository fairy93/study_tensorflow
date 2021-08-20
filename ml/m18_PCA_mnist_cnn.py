import numpy as np
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.python.keras.layers.pooling import MaxPooling1D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.decomposition import PCA

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000, 28*28)

pca = PCA(n_components=400) 
x = pca.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)
# print(x_train.shape, x_test.shape)# (56000, 400) (14000, 400)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 20, 20)
x_test = x_test.reshape(x_test.shape[0], 20, 20)

onehot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = onehot.fit_transform(y_train)
y_test = onehot.transform(y_test)

#2. 모델
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,
          padding='same', input_shape=(20, 20)))
model.add(MaxPooling1D())
model.add(Conv1D(30, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#3. 컴파일 구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000,
          batch_size=256, verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time()-start_time


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss', loss[0])
print('acc', loss[1])


# cnn
# time 32.76616168022156
# loss 0.06886795163154602
# acc 0.9819999933242798

# cnn - PCA
# time 56.29845833778381
# loss 0.1915593296289444
# acc 0.9457142949104309