import numpy as np
from tensorflow.python.keras.layers.core import Flatten
# 실습
# 1~100까지의 데이터를

#       X           Y
# 1,2,3,4,5         6
# ....
# 95,96,97,98,99   100


x_data = np.array(range(1, 101))
x_predict = np.array(range(96,106))

#           x
# 96,97,98,99,1100          ?
# ....
# 101,102,103,104,105,106   ?
# 예상 결과값 : 10 102 013 014 105 106
size =6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)

print(dataset) 

# x = dataset[:, :5].reshape(95,5,1)
x = dataset[:, :5y = dataset[:, 5]

# print("x : ", x) 
# print("y : ", y) 

#모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D

model = Sequential()
# model.add(LSTM(64, input_shape=(5,1)))
model.add(Conv1D(64,2,input_shape=(5,1)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()