from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape

model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(28, 28)))
model.add(Flatten())  # (n,780)
model.add(Dense(784))  # (n,784)
model.add(Reshape(28, 28, 1))
model.add(Conv2D(64, (2, 2)))
model.add(Conv2D(64, (2, 2)))
model.add(Conv2D(64, (2, 2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(10, activation='softmax'))
model.summary()
