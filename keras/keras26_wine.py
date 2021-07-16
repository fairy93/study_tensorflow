from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_wine

#완성하시오
# acc0.8.이상만들것
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=70)


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler= MaxAbsScaler()
# scaler= RobustScaler()
# scaler= QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
input1 = Input(shape=(13,))
dense1 = Dense(32, activation='relu',name='dense1')(input1)
dense2 = Dense(64, activation='relu',name='dense2')(dense1)
dense3 = Dense(128, activation='relu',name='dense3')(dense2)
dense4 = Dense(64, activation='relu',name='dense4')(dense3)
dense5 = Dense(32, activation='relu',name='dense5')(dense4)
dense6 = Dense(16, activation='relu',name='dense6')(dense5)
dense7 = Dense(8, activation='relu',name='dense7')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일구현
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=4,validation_split=0.2,verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss',loss)
y_predict = model.predict(x_test)
# print('예측값 ',y_predict)

#5. r2 예측r2
r2= r2_score(y_test,y_predict)
print(r2)

# #결과 07.15
# loss 0.03506023436784744
# r2 0.9444673310786067