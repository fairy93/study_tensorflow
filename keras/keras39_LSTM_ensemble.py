from tensorflow.keras.layers import concatenate, Concatenate
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU
from numpy import array
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], [20, 30, 40],
            [30, 40, 50], [40, 50, 60]])
x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60],
            [30, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
            [90, 100, 110], [100, 110, 120],
            [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
# (batch_size, timesteps, feature)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
x1_predict = x1_predict.reshape(1, x1_predict.shape[0], 1)
x2_predict = x2_predict.reshape(1, x2_predict.shape[0], 1)
print(x1.shape, y.shape)


#2. 모델 구성
#2-1. 모델1
input1 = Input(shape=(3, 1))
lstm = LSTM(units=32, activation='relu', input_shape=(3, 1))(input1)
dense1 = Dense(64, activation='relu')(lstm)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
output1 = Dense(8)(dense4)

#2-1. 모델2
input2 = Input(shape=(3, 1))
lstm = LSTM(units=33, activation='relu', input_shape=(3, 1))(input2)
dense11 = Dense(10, activation='relu')(lstm)
dense12 = Dense(10, activation='relu')(dense11)
dense13 = Dense(10, activation='relu')(dense12)
dense14 = Dense(10, activation='relu')(dense13)
output2 = Dense(8, name='output2')(dense14)


        # merge1 = concatenate([output1, output2])
        # merge2 = Dense(10)(merge1)
        # merge3 = Dense(5, activation='relu')(merge2)

        # last_output = Dense(1)(merge3)
        # model = Model(inputs=[input1, input2], outputs=last_output)


# #3. 컴파일 구현
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
# model.fit([x1, x2], y, epochs=1000, batch_size=32, verbose=1, callbacks=[es])


# #4. 평가, 예측
# loss = model.evaluate([x1, x2], y)
# res = model.predict([x1_predict, x2_predict])
# print('loss: ', loss)
# print('res', res)

# # 결과 210722
# # loss:  6.371002382365987e-05
# # res [[86.50015]]
