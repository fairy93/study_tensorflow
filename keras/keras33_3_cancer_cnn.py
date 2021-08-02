from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2),
          padding='same', input_shape=(x_train.shape[1], 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일구현
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # loss metrics
print('loss', loss[0])
print('acc : ', loss[1])


# # 결과 2021.07.16
# # epochs=1000,batch_size=32,validation_batch_size=0.2
# # loss (binary_crossentropy) 0.4764081835746765
# # acc :  0.9532163739204407

# 결과 cnn 21.07.22
# loss 0.1878443956375122
# acc :  0.9298245906829834
