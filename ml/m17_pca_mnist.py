import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test,_) = mnist.load_data() # _ 의미머야 ?

print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train,x_test,axis=0)
print(x.shape) # (70000, 28, 28)