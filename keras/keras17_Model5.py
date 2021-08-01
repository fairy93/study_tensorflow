from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(
    1, 101), range(100), range(401, 501)])
x = np.transpose(x)

# print(x.shape) # shape
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
# print(y.shape) # shape

#2. 모델구성 input_shape=(5,), out_put=2
input1 = Input(shape=(5,))  # input layer modeling
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()
