from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # x_train = (60000, 28, 28) / y_train = (60000,)
print(x_test.shape, y_test.shape) # x_test = (10000, 28, 28) / y_test = (10000,)

print(x_train[120])
print("y_train[120] 값 : ", y_train[120])

plt.imshow(x_train[120], 'gray')
plt.show()