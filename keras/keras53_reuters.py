from numpy import testing
from tensorflow.keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

(x_train, y_train),(x_test, y_test)= reuters.load_data(num_words=10000, test_split=0.2)

print(x_train[0], type(x_train[0]))
print(y_train[0]) # 3

print(len(x_train[0]),len(x_train[1])) #87 56

# 판다스 넘파이 shape 가능
# 리스트길이알려면 len, numpy로 변환

print(x_train.shape,x_test.shape)
print(y_train.shape, y_test.shape)

print(type(x_train)) # <class 'numpy.ndarray'>
print('뉴스기사의 최대길이 ',max(len(i) for i in x_train))
print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train))

# plt.hist([len(s) for s in x_train],bins=50)
# plt.show()


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100,padding='pre')
print(x_train.shape, x_test.shape)
print(type(x_train),type(x_train[0])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(x_train[0])
print(x_train[1])

print(np.unique(y_train))

# 10개중에하나 5개중하나 softmax 카테고르엔트로피 ..

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(*y_train.shape, y_test.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

#실습 완성해보세요