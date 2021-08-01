from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import time

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1.1, 1.2, 1.3, 1.4,
             1.5, 1.6, 1.5, 1.4, 1.3], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])  # (3,10)
x = np.transpose(x)
print(x.shape)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.transpose(y)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
start = time.time()
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=10, verbose=3)
end = time.time()-start
print(end)

# verbose = 0
# 결과만 보여줘 처리시간
# time = 14.904160261154175

# verbose = 1
# default
# time = 21.22871446609497

# verbose = 2
# epochs+loss
# time = 16.97713875770569

# verbose = 3
# epochs
# time = 16.632190942764282

# verbos = 1 일때
# batch=1, 10인 경우 시간측정

# #결과 21.07.14
# verbos = 1 일때,batch_size=1,10 의 시간측정
# batch_size=1
# time=20.970410108566284
# batch_size=10
# time=3.662076473236084
