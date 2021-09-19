from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.ops.array_ops import sequence_mask

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
# (150, 4) (150,)

y=to_categorical(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=70)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(100,input_dim=4))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
es = EarlyStopping(monitor='val_loss',patience=50,mode='min')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=2000,batch_size=16,validation_split=0.2,callbacks=[es],verbose=2)

#4. 평가
loss = model.evaluate(x_test,y_test)
print('loss',loss[0])
print('acc',loss[1])


# 결과 2021.07.16
# epochs=1000,batch_size=8,validation_batch_size=0.2
# loss (binary_crossentropy) 0.4764081835746765
# loss 0.2432488352060318
# accuracy :  0.9777777791023254

# 결과 2021.09.16
# epochs=2000,batch_size=16,validation_batch_size=0.2
# loss 0.00018315248598810285
# acc 1.0
