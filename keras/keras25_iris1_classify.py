import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape) #(150,4) (150,)
# # print(y)
# # 원핫인코딩 one-hot-encoding #(150,) -> (150,3)
# # 0->[1,0,0]
# # 1->[0,1,0]
# # 2->[0,0,1]

# # 0,1,2,1
# # [[1,0,0]
# # [0,1,0]
# # [0,0,1]
# # [0,1,0]] (4,) ->(4,3)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=70)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
hist = model.fit(x_train,y_train,epochs=1000,batch_size=8,validation_batch_size=0.2, callbacks=[es])



#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # loss metrics
print('loss',loss[0])
print('accuracy : ',loss[1])
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)


# 결과 2021.07.16
# epochs=1000,batch_size=8,validation_batch_size=0.2
# loss (binary_crossentropy) 0.4764081835746765
loss 0.2432488352060318
accuracy :  0.9777777791023254