import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes  # 당뇨

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (442, 10), (442,)

# print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(datasets.DESCR)
print(y[:30])
print(np.min(y), np.max(y))

#2. 모델 구성
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1004)
model = Sequential()
model.add(Dense(5, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu'))  # 활성화함수
model.add(Dense(100, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=120, batch_size=1, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)
print('예측값 ', y_predict)

#5. r2 예측r2
r2 = r2_score(y_test, y_predict)
print(r2)
