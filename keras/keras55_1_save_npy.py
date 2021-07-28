from sklearn import datasets
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer, load_wine
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
import numpy as np

# iris npy save
datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data)

# boston npy save
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data)


# diabetes npy save
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
np.save('./_save/_npy/k55_x_data_diabetes.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_diabetes.npy', arr=y_data)

# breast_cancer npy save
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
np.save('./_save/_npy/k55_x_data_breast_cancer.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_breast_cancer.npy', arr=y_data)


# wine_cancer npy save
datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data)

# mnist npy save
(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test)

#fashion_mnist npy save
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.save('./_save/_npy/k55_x_train_fashion_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_fashion_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_fashion_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_fashion_mnist.npy', arr=y_test)

# cifar10 npy save
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=y_test)

# cifar100 npy save
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test)