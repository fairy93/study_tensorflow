from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten, Conv1D,LSTM

#1 데이터
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=10000)

print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape) # (25000,) (25000,)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
x_train = pad_sequences(x_train, padding='pre', maxlen=3000) # (25000, 200) 
x_test = pad_sequences(x_test, padding='pre', maxlen=3000) # (25000, 200)



y_train = to_categorical(y_train) # (25000, 2) 
y_test = to_categorical(y_test) # (25000, 2)
print(y_train.shape, y_test.shape)


#2 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=3000))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

#3 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=512, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time


#4 평가예측
acc = model.evaluate(x_test, y_test)[1]
print("time : ", end_time)
print('acc : ', acc)

# time :  85.60995507240295
# acc :  0.8332800269126892