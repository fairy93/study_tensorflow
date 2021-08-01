from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=80)

#2. 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=1))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3400, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # 평가
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predic 의 값은 ', y_predict)

r2 = r2_score(y_test, y_predict)
print(r2)


# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=3400, batch_size=1
# loss :  8.06191294278058e-11
# r2 : 0.99999999999989
