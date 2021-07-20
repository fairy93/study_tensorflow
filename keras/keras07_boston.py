from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

datasets = load_boston()
x=datasets.data
y=datasets.target

datasets.feature_names
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True,random_state=19)

#2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))

#3. 컴파일구현
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=3000,batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss',loss)

y_predict = model.predict(x_test)
print('예측값 ',y_predict)

#5. r2 예측r2
r2= r2_score(y_test,y_predict)
print(r2)

# #결과 21.07.14
# loss='mse', optimizer='adam'
# epochs=1500, batch_size=1
# loss :  29.254825592041016
# r2 : 0.6149948921834689

# #결과 21.07.20
# loss='mse', optimizer='adam'
# epochs=3000, batch_size=32
# loss 17.916685104370117
# r2 : 0.8287657927527008