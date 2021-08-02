import datetime
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model, load_model
from operator import concat
import numpy as np
from sklearn.metrics import r2_score
from scipy.sparse.construct import rand
from tensorflow.python.keras.layers.core import Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
# y = np.array([range(1001, 1101)])
# y = np.transpose(y1)
y = np.array(range(1001, 1101))

# print(x1.shape, x2.shape, y.shape)  # (100, 3) (100, 3) (100,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
                                                                         train_size=0.7, shuffle=True, random_state=66)

# print(x1_train.shape)  # (70, 3)


# 모델구성

# 모델1
input1 = Input(shape=(3, ))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(3, name='output1')(dense3)

# 모델2
input2 = Input(shape=(3, ))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(4, name='output2')(dense14)


model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()
#merge1 = concatenate([output1, output2])
merge1 = Concatenate()([output1, output2])
merge2 = Dense(10)(merge1)
lastoutput = Dense(1)(merge2)

model = Model(inputs=[input1, input2], outputs=lastoutput)
model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=20,
                   mode='auto', verbose=1, restore_best_weights=False)
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath = './_save/ModelCheckPoint/'
filename = '.{epoch:04d}--{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k47_", date_time, "_", filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=filepath+filename)
model.fit([x1_train, x2_train], y_train, epochs=100,
          batch_size=32, verbose=2, validation_split=0.2, callbacks=[es, mcp])

model.save('./_save/ModelCheckPoint/keras47_model_save.h5')


print("=========================1. 기본출력===========================")
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss: ', loss[0])

y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2', r2)

print("=========================2. load model ===========================")
model2 = load_model('./_save/ModelCheckPoint/keras47_model_save.h5')
loss = model2.evaluate([x1_test, x2_test], y_test)
print('loss: ', loss)

y_predict = model2.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2', r2)

# print("=========================3. model Check Point ===========================")
# model3 = load_model('./_save/ModelCheckPoint/keras49_MCP.h5')
# loss = model3.evaluate([x1_test, x2_test], y_test)
# print('loss: ', loss)

# y_predict = model3.predict([x1_test, x2_test])
# r2 = r2_score(y_test, y_predict)
# print('r2', r2)
