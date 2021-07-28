import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=60)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler= MaxAbsScaler()
# scaler= RobustScaler()
# scaler= QuantileTransformer()
scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test= scaler.transform(x_test)

#2. 모델 구성
input1 = Input(shape=(10,))
dense1 = Dense(32, activation='relu',name='dense1')(input1)
dense2 = Dense(64, activation='relu',name='dense2')(dense1)
dense5 = Dense(32, activation='relu',name='dense5')(dense2)
dense6 = Dense(16, activation='relu',name='dense6')(dense5)
dense7 = Dense(8, activation='relu',name='dense7')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
import datetime
date = datetime.datetime.now()
date_time=date.strftime("%m%d_%H%M")
filepath = './_save/ModelCheckPoint/'
filename = '.{epoch:04d}--{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k48_2",date_time,"_",filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=filepath+filename)

model.fit(x_train,y_train,epochs=200,batch_size=32,validation_split=0.2,verbose=2,callbacks=[es,mcp])
model.save('./_save/ModelCheckPoint/keras48_2_model_save.h5')

#4. 평가, 예측
print("=========================1. 기본출력===========================")

loss = model.evaluate(x_test, y_test)
print('loss',loss)
y_predict = model.predict(x_test)
r2= r2_score(y_test, y_predict)
print('r2',r2)

print("=========================2. load model ===========================")
model2 = load_model('./_save/ModelCheckPoint/keras48_2_model_save.h5')
loss = model2.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2', r2)




