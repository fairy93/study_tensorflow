from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
import numpy as np

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  # (150,4) (150,)
# # print(y)
# # 원핫인코딩 one-hot-encoding #(150,) -> (150,3)
# # 0->[1,0,0]
# # 1->[0,1,0]
# # 2->[0,0,1]

# # 0,1,2,1
# # [[1,0,0]
# # [0,1,0]
# # [0,0,1]
# # [0,1,0]] (4,) ->(4,3)
y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=70)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성

input1 = Input(shape=(4, 1))
lstm = LSTM(units=256, activation='relu')(input1)
dense1 = Dense(256, activation='relu')(lstm)
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(64, activation='relu')(dense4)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
output1 = Dense(3,activation="softmax")(dense7)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일구현
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
                 validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # loss metrics
print('loss', loss[0])
print('acc : ', loss[1])

# 결과 210722
# loss 0.08977647125720978
# acc :  0.9777777791023254
