from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as py

#1. 데이터
datasets=load_boston()
x=datasets.data
y=datasets.target

x_train, x_test, ytrain, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=1004)

#2. 모델 구상
input1 = Input(shape=(13,))
dense1 = Dense(5)(input1)
dense2 = Dense(100)(dense1)
dense3 = Dense(90)(dense2)
dense4 = Dense(110)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)
model.summary()