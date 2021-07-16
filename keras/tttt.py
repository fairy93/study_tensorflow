from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
datasets = load_boston()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=1004)

print(np.min(x),np.max(x))

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

from keras.callbacks import History
#3. 컴파일구현
model.compile(loss='mse', optimizer='adam',validation)
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=5,mode='min',verbose=1)
hist = model.fit(x_train,y_train,epochs=10,batch_size=8,validation_batch_size=0.2, callbacks=[es])

print(hist)
print(hist.history.keys())
# print('--------------------------------')
# print(hist.histroy['loss'])
# print('---------------------------------')
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title("loss, val_loss")
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
plt.legend(['train loss','val loss'])
# plt.show()

# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test)
# print('loss',loss)
# y_predict = model.predict(x_test)
# print('예측값 ',y_predict)



# #5. r2 예측r2
# r2= r2_score(y_test,y_predict)
# print(r2)