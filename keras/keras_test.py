import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train),(x_test,y_test) = mnist.load_data()

# x_train(60000, 28, 28) y_train(60000,)
# x_test(10000, 28, 28) y_test(10000,)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

onhot = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = onhot.fit_transform(y_train)
y_test = onhot.transform(y_test)

print(y_train.shape)
# model = Sequential()
# model.add(Conv2D(100,kernel_size=(2,2),padding='same',input_shape=(28,28,1)))
# model.add(Conv2D(100,(2,2)))