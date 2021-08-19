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
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
# model = XGBRegressor()
model = RandomForestRegressor()
# model = GradientBoostingRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
result = model.score(x_test, y_test)
print('결과 : ', result)

# XGBRegressor
# 결과 :  0.3210924574289413

# RandomForestRegressor
# 결과 :  0.RandomForestRegressor

# GradientBoostingRegressor
# 결과 :  0.415118544943054
