
import numpy as np
from scipy.sparse.construct import rand
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv',sep=';',index_col=None, header=0)

# print(datasets)
# print(datasets.shape) #(4898,12)
# print(datasets.info())
# print(datasets.describe())

data = datasets.to_numpy()
x=data[:,:11]
y=data[:,11:]

# print(x.shape,y.shape)
# print(np.unique(y))
onehot = OneHotEncoder(sparse=False)
onehot.fit(y)
y=onehot.transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.7,shuffle=True,random_state=33)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model=Sequential()
model.add(Dense(64,input_shape=(11,)))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(7,activation='softmax'))

#3. 컴파일 구현
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time=time.time()
model.fit(x_train,y_train, epochs=1000, batch_size=16,validation_split=0.2,callbacks=[es])
end_time=time.time()-start_time

#4. 평가 예측
loss =model.evaluate(x_test,y_test)
print('time',end_time)
print('loss',loss[0])
print('acc',loss[1])

# epochs=2000, batch_size=8
# loss :  1.0809568166732788
# acc :  0.539455771446228

# epochs=2000, batch_size=2
# loss :  1.0815858840942383
# acc :  0.543537437915802

# epochs=1000, batch_size=16
# time 20.10618805885315
# loss 1.1205155849456787
# acc 0.5571428537368774