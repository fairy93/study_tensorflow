import time

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=77)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

#2. 모델
model = Sequential()
model.add(Conv2D(filters=100,kernel_size=(2,2),padding='same',input_shape=(13,1,1)))
model.add(Conv2D(80, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(80, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(80, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(80, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1))

#3. 컴파일 구현
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='min')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
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



# dnn
# loss 17.916685104370117
# r2 : 0.8287657927527008

# cnn 21.07.21
# time 20.870214700698853
# loss 18.926462173461914
# r2 0.8046224569655018

# cnn 21.09.29
# time 15.198190212249756
# loss 13.334495544433594
# r2 0.8076723783203633