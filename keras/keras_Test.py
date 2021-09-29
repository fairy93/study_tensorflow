import time

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=67)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

#2. 모델
model = Sequential()
model.add(Conv2D(filters=100,kernel_size=(2,2),padding='same',input_shape=(x_train.shape[1],1,1)))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(80, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(80, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1))


#3. 컴파일 구현
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.1, verbose=2, callbacks=[es])
end_time = time.time()-start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#5. r2
r2 = r2_score(y_test, y_predict)

print('time', end_time)
print('loss', loss)
print('r2', r2)


time 21.033281564712524
loss 2841.651611328125
r2 0.42989599458282846

