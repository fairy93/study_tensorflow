import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test,y_test)=mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape,y_test.shape)

print(x_train[120])
print("y[0] 값",y_train[3000])

plt.imshow(x_train[3000],'gray')
plt.show()