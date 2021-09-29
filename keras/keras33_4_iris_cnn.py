import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=57)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

#2. 모델
model = Sequential()
model.add(Conv2D(filters=100,kernel_size=(2,2),padding='same',input_shape=(x_train.shape[1],1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))


#3. 컴파일 구현
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=30, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=64,
          validation_split=0.1, verbose=2, callbacks=[es])
end_time = time.time()-start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # loss metrics
print('time',end_time)
print('loss', loss[0])
print('acc : ', loss[1])



# 결과 dnn 2021.07.16 dnn
# epochs=1000,batch_size=8,validation_split=0.2
# loss (binary_crossentropy) 0.4764081835746765
# loss 0.2432488352060318
# acc 0.9777777791023254

# 결과 cnn 21.07.22
# loss 0.09633602201938629
# acc:  0.9555555582046509

# 결과 cnn 21.09.29
# time 1.838169813156128
# loss 0.1889994740486145
# acc :  0.9666666388511658