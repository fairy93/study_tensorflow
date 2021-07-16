from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential() #(N,5,5,1)
model.add(Conv2D(10,kernel_size=(2,2), padding='same', input_shape=(5,5,1))) #(N,4,4,10) 이미지 통과시켜도 4차원 #paddig same 하면 5.5.10 으로전달
model.add(Conv2D(20,(2,2), activation='relu')) #(N,3,3,20)           # padding defalut = valid
model.add(Conv2D(30,(2,2), padding='valid'))
model.add(Flatten()) #(N, 180) 여
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()