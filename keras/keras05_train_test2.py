from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
# 모델엔 2가지 있어 순차적모델, 함수형 모델

#1. 데이터
# 과적합? 내신잘봐봣자 수능못보면 소요없어
# 훈련용 테스트 데이터 구분
x = np.array(range(100))
y = np.array(range(1, 101))


x_train = x[:70]
y_train = y[:70]
x_test = x[-30:]
y_test = y[-30:]
