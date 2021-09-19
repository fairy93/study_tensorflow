from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=67)

scaler= StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(100,input_dim =30))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss',patience=30, mode='min')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=1000,batch_size=32,validation_split=0.1,callbacks=[es],verbose=2)

#4. 평가 
loss =model.evaluate(x_test,y_test)
print('loss ',loss[0])
print('acc ',loss[1])


# 결과 2021.07.16
# epochs=1000,batch_size=8,validation_batch_size=0.2
# loss (binary_crossentropy) 0.4764081835746765
# accuracy :  0.9532163739204407

# 결과 2021.09.16
# epochs=1000,batch_size32,validation_batch_size=0.1
# loss  0.1579935997724533
# acc  0.9736841917037964