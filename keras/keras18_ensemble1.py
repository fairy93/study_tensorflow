from tensorflow.keras.layers import concatenate, Concatenate
from operator import concat
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

x1 = np. array([range(100), range(301, 401), range(1, 101)])
x2 = np .array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001, 1101))

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, shuffle=True, random_state=1004)
# train_size를 넣지않으면 default=0.7
# print(x1_train.shape)
# print(x1_test.shape)
# print(x2_train.shape)
# print(x2_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#2. 모델구성


#2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(1, name='output1')(dense3)

#2-1. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(1, name='output2')(dense14)

# concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)

last_output = Dense(1)(merge3)
model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=2)
res = model.evalute([x1_test, x2_test], y_test)
