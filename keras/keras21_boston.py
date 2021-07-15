from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input
from matplotlib import pyplot as plt
import numpy as np
datasets = load_boston()
x=datasets.data
y=datasets.target

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

# #결과 21.07.15
# scaler= MaxAbsScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 13.372722625732422
# r2 0.8619536023526895

# #결과 21.07.15
# scaler = StandardScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 11.2092866897583
# r2 0.8842867254686915

# 결과 21.07.15
# scaler= MaxAbsScaler() 
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 13.744830131530762
# r2 0.8581123591793408

# 결과 21.07.15
# scaler= RobustScaler()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 13.318341255187988
# r2 0.8625149788237876

# 결과 21.07.15
# scaler= QuantileTransformer()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 19.03533363342285
# 42 0.8034985673302772


# 결과 21.07.15
# scaler = PowerTransformer()
# loss='mse', optimizer='adam'
# epochs=1000, batch_size=4, validation_split=0.2
# loss 12.240410804748535
# 0.8736424397988549

