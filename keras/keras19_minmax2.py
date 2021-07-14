from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
datasets = load_boston()
x=datasets.data
y=datasets.target

print(np.min(x),np.max(x))
# x= x/711. 소수점 ?? 
# x = x/np.min(x)
# x= (x-np.min(x))/(np.max(x)-np.min(x))
# x=x/np.max(x)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_scale = scaler.transform(x)
print(x_scale[:10])
print(np.min(x_scale),np.max(x_scale))

x_train, x_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.3,shuffle=True,random_state=1004)


#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일구현
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=25000,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss',loss)
y_predict = model.predict(x_test)
print('예측값 ',y_predict)



#5. r2 예측r2
r2= r2_score(y_test,y_predict)
print(r2)