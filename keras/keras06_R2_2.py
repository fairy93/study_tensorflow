from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np

# 출력 로스값
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])
x_pred = [6]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10000, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x, y)  # 평가
print('loss : ', loss)

y_predict = model.predict(x)
print('x_pred의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)
print(r2)


# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=1500, batch_size=1
# loss :  0.38063496351242065
# r2 : 0.8096825191879915

# #결과 21.07.20
# loss='mse', optimizer='adam'
# epochs=3000, batch_size=32
# loss :  0.37999993562698364
# r2 : 0.8100000190734746
