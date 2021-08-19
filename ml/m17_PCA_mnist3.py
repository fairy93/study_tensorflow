import numpy as np
import time

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000, 28*28)

pca = PCA(n_components=486)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=66)

onehot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot.fit(y_train)
y_train = onehot.fit_transform(y_train)
y_test = onehot.fit_transform(y_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(100, input_shape=(486,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss', patience=15, mode='min')
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
          validation_split=0.2, verbose=2, callbacks=[es])
end_time = time.time()-start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('time', end_time)
print('loss', loss[0])
print('acc', loss[1])

# 결과 dnn 210721
# time 31.354090690612793
# loss 0.0970178171992302
# acc 0.9783999919891357

# 결과 cnn 210721
# time 51.061420917510986
# loss 0.07350124418735504
# acc 0.9868000149726868

# 결과 pca = PCA(n_components=486)
# time 118.66610980033875
# loss 0.17424745857715607
# acc 0.9481428861618042
