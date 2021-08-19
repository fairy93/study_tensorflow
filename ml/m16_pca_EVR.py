import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

#1. 데이터
datasets = load_diabetes()
x = datasets.data  # (442, 10)
y = datasets.target  # (442,)

pca = PCA(n_components=7)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
#print(pca_EVR) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
# print(sum(pca_EVR)) # 0.9479436357350414

cumsum = np.cumsum(pca_EVR)
# print(cumsum) # 누적합
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

# print(np.argmax(cumsum >= 0.94)+1) # 7

plt.plot(cumsum)
plt.grid()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66)

# 2. 모델
model = XGBRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
result = model.score(x_test, y_test)
print('결과 : ', result)

# PCA(n_components=7)
# 결과 :  0.3210924574289413

# PCA(n_components=6)
# 결과 :  0.29062991982314124
