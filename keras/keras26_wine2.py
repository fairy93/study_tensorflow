
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

#1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv',sep=';',index_col=None, header=0)

# print(datasets)
print(datasets.shape) #(4898,12)
print(datasets.info())
print(datasets.describe())

# 다중분류
# 모델링하고 
# 0.8이상완성

data=datasets.to_numpy()
x=data[:,:11]
y=data[:,11]
print(x.shape,y.shape)
print(np.unique(y))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(y)
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=70)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)
# x_train= scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# # array([[1., 0., 0.,0.,0.,0.,0.],[[0., 1., 0.,0.,0.,0.,0.], [[0., 0., 1.,0.,0.,0.,0.], [[0., 0., 0.,1.,0.,0.,0.], [[0., 0., 0.,0.,1.,0.,0.], 
# # [[0., 0., 0.,0.,0.,1.,0.], [[0., 0., 0.,0.,0.,0.,1.])


# #2. 모델 구성
# model = Sequential()
# model.add(Dense(128, activation='relu', input_dim=11))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# #3. 컴파일구현
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
# model.fit(x_train,y_train,epochs=1000,batch_size=32,validation_split=0.2, callbacks=[es])

# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test) # loss metrics
# print('loss',loss[0])
# print('accuracy : ',loss[1])


# # # # y_predict = model.predict(x_test)
# # # # print('예측값 ',y_predict)

# # # # #5. r2 예측r2
# # # # r2= r2_score(y_test,y_predict)
# # # # print(r2)
