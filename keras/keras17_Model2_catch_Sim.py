#06_r2-2를 카피
# 함수형으로 리폼하시오
#서머리로 확인
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred=[6]

#2. 모델 구성
input1 = Input(shape=(1,))
dense1 = Dense(1)(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(12)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()