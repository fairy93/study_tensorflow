import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

x_train = x[:7]
y_train = y[:7]
x_test = x[7:10]
y_test = y[7:10]
x_val = x[10:]
y_val = y[10:]

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)
