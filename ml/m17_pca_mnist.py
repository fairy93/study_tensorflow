import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()  # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000, 28*28)

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# print(cumsum)

print(np.argmax(cumsum >= 1.00)+1)

# 0.95 / 154
# 0.99 / 331
# 1.00 / 713
