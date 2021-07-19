from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # x_train = (50000, 32, 32, 3) / y_train = (50000, 1)
print(x_test.shape, y_test.shape) # x_test = (10000, 32, 32, 3) / y_test = (10000, 1)

print(x_train[23])
print("y_train[20] 값 : ", y_train[23])

plt.imshow(x_train[23], 'gray')
plt.show()