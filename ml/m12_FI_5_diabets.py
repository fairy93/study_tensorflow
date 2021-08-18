# feature_importance를 돌려 데이터가 20~25%미만인 데이터를 지우고 데이터를 재구성 한뒤
# 모델별로 결과 구하기

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)
#           age       sex       bmi        bp        s1        s2        s3        s4        s5        s6
# 0    0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019908 -0.017646
# 1   -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068330 -0.092204
# 2    0.085299  0.050680  0.044451 -0.005671 -0.045599 -0.034194 -0.032356 -0.002592  0.002864 -0.025930
# 3   -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022692 -0.009362
# 4    0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031991 -0.046641
# ..        ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
# 437  0.041708  0.050680  0.019662  0.059744 -0.005697 -0.002566 -0.028674 -0.002592  0.031193  0.007207
# 438 -0.005515  0.050680 -0.015906 -0.067642  0.049341  0.079165 -0.028674  0.034309 -0.018118  0.044485
# 439  0.041708  0.050680 -0.015906  0.017282 -0.037344 -0.013840 -0.024993 -0.011080 -0.046879  0.015491
# 440 -0.045472 -0.044642  0.039062  0.001215  0.016318  0.015283 -0.028674  0.026560  0.044528 -0.025930
# 441 -0.045472 -0.044642 -0.073030 -0.081414  0.083740  0.027809  0.173816 -0.039493 -0.004220  0.003064

df.drop(['s4', 's1', 'sex'], inplace=True, axis=1)
x = df.to_numpy()
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=44)

#2. 모델
model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = DecisionTreeRegressor()
# model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
r2 = model.score(x_test, y_test)
print('r2 : ', r2)

print(model.feature_importances_)

# RandomForestRegressor
# df.drop(['s4','s1','sex'], inplace=True, axis=1)
# r2 :  0.3688913037278112
# [0.07751889 0.2567734  0.08645301 0.08683204 0.06227974 0.35056035
#  0.07958257]

# GradientBoostingRegressor
# df.drop(['s4','s1','sex'], inplace=True, axis=1)
# r2 :  0.38814930339615694
# [0.06856764 0.28623467 0.08243698 0.08940384 0.04162995 0.37376151
#  0.05796541]

# DecisionTreeRegressor
# df.drop(['s4','s1','sex'], inplace=True, axis=1)
# r2 :  -0.41289564746675467
# [0.08906296 0.24129862 0.07344866 0.1066189  0.04017065 0.36886745
#  0.08053276]

# XGBRegressor
# df.drop(['s4','s1','sex'], inplace=True, axis=1)
# r2 :  0.20359464905750557
# [0.04610109 0.17556281 0.06981704 0.09149326 0.06869822 0.45614192
#  0.09218565]
