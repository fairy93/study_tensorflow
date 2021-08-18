# feature_importance를 돌려 데이터가 20~25%미만인 데이터를 지우고 데이터를 재구성 한뒤
# 모델별로 결과 구하기

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)
#         CRIM    ZN  INDUS CHAS    NOX     RM   AGE     DIS  RAD    TAX PTRATIO       B LSTAT
# 0    0.00632  18.0   2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0    15.3  396.90  4.98
# 1    0.02731   0.0   7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0    17.8  396.90  9.14
# 2    0.02729   0.0   7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0    17.8  392.83  4.03
# 3    0.03237   0.0   2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0    18.7  394.63  2.94
# 4    0.06905   0.0   2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0    18.7  396.90  5.33
# ..       ...   ...    ...  ...    ...    ...   ...     ...  ...    ...     ...     ...   ...
# 501  0.06263   0.0  11.93  0.0  0.573  6.593  69.1  2.4786  1.0  273.0    21.0  391.99  9.67
# 502  0.04527   0.0  11.93  0.0  0.573  6.120  76.7  2.2875  1.0  273.0    21.0  396.90  9.08
# 503  0.06076   0.0  11.93  0.0  0.573  6.976  91.0  2.1675  1.0  273.0    21.0  396.90  5.64
# 504  0.10959   0.0  11.93  0.0  0.573  6.794  89.3  2.3889  1.0  273.0    21.0  393.45  6.48
# 505  0.04741   0.0  11.93  0.0  0.573  6.030  80.8  2.5050  1.0  273.0    21.0  396.90  7.88

df.drop(['ZN', 'INDUS', 'CHAS', 'AGE', 'RAD', 'TAX', 'B'], inplace=True, axis=1)
x = df.to_numpy()
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = DecisionTreeRegressor()
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
r2 = model.score(x_test, y_test)
print('r2 : ', r2)

print(model.feature_importances_)

# RandomForestRegressor
# df.drop([ 'ZN','INDUS','CHAS','AGE' ,'RAD' ,'TAX' , 'B' ], inplace=True, axis=1)
# r2 :  0.9110644644539202
# [0.05099089 0.02987325 0.40557994 0.0752487  0.02447415 0.41383307]

# GradientBoostingRegressor
# df.drop([ 'ZN','INDUS','CHAS','AGE' ,'RAD' ,'TAX' , 'B' ], inplace=True, axis=1)
# r2 :  0.9311174618255292
# [0.02772389 0.04540356 0.36462298 0.09027199 0.03638204 0.43559553]

# DecisionTreeRegressor
# df.drop([ 'ZN','INDUS','CHAS','AGE' ,'RAD' ,'TAX' , 'B' ], inplace=True, axis=1)
# r2 :  0.8078109479877618
# [0.05096627 0.02314231 0.28239041 0.06675814 0.0356029  0.54113997]

# XGBRegressor
# df.drop([ 'ZN','INDUS','CHAS','AGE' ,'RAD' ,'TAX' , 'B' ], inplace=True, axis=1)
# r2 :  0.917935754624127
# [0.01795975 0.05797544 0.3119248  0.06741601 0.0472353  0.49748865]
