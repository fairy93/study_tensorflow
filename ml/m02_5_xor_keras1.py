import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 컴파일 구현
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가 예측
y_pred = model.predict(x_data)
print(y_pred)
y_pred = np.round(y_pred)
print(x_data, 'result :', y_pred)

acc = accuracy_score(y_data, y_pred)
print('acc_score : ', acc)

# [[0, 0], [0, 1], [1, 0], [1, 1]] result : [[0.][1.][1.][1.]]
# acc_score :  0.75
