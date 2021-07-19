
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv',sep=';',index_col=None, header=0)

# print(datasets)
# print(datasets.shape) #(4898,12)
# print(datasets.info())
# print(datasets.describe())


data=datasets.to_numpy()
x=data[:,:11]
y=data[:,11:]
print(x.shape,y.shape)
print(np.unique(y))


oneHot_encoder = OneHotEncoder(sparse=False)  
y = oneHot_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=70)

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
input1 = Input(shape=(11,))
dense1 = Dense(100, activation='relu', name='dense1')(input1)
dense2 = Dense(100, activation='relu', name='dense2')(dense1)
dense3 = Dense(100, activation='relu', name='dense3')(dense2)
dense4 = Dense(100, activation='relu', name='dense4')(dense3)
dense5 = Dense(100, activation='relu', name='dense5')(dense4)
output1 = Dense(7, activation='softmax', name='output1')(dense5)

model = Model(inputs= input1, outputs=output1)

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
model.fit(x_train, y_train, epochs=2000, batch_size=2, callbacks=[es], validation_split=0.2, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

# epochs=2000, batch_size=8
# loss :  1.0809568166732788
# acc :  0.539455771446228

# epochs=2000, batch_size=2
# loss :  1.0815858840942383
# acc :  0.543537437915802