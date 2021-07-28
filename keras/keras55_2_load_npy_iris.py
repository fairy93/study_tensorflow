import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

x_data=np.load('./_save/_npy/k55_x_data_iris.npy')
y_data=np.load('./_save/_npy/k55_y_data_iris.npy')

y_data = to_categorical(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.3,shuffle=True,random_state=70)


scaler = StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=4))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일구현
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
hist = model.fit(x_train,y_train,epochs=1000,batch_size=8,validation_split=0.2, callbacks=[es])



#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # loss metrics
print('loss',loss[0])
print('accuracy : ',loss[1])