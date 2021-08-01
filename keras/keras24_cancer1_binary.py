from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

datasets = load_breast_cancer()


# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=70)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=30))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일구현
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
                 validation_batch_size=0.2, callbacks=[es])

# print(hist)
# print(hist.history.keys())
# print('--------------------------------')
# print(hist.history['loss'])
# print('---------------------------------')
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title("loss, val_loss")
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss','val loss'])
# plt.show()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # loss metrics
print('loss', loss[0])
print('accuracy : ', loss[1])

loss = model.evaluate(x_test, y_test)  # loss metrics
print(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
print(y_predict)

# # 결과 2021.07.16
# # epochs=1000,batch_size=8,validation_batch_size=0.2
# # loss (binary_crossentropy) 0.4764081835746765
# # accuracy :  0.9532163739204407
