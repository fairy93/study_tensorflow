import time

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_wine()
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



# 결과 07.21
# epochs=100, batch_size=8
# time 2.4942734241485596
# loss  0.0605735220015049
# acc 0.9814814925193787

# 결과 cnn 07.21
# epochs=100, batch_size=8
# time 6.185149908065796
# loss  0.3106430172920227
# acc 0.9259259104728699

# 결과 cnn 09.29
# time 6.408697605133057
# loss 0.10609585791826248
# acc :  0.9722222089767456