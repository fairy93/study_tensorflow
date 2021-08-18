# feature_importance를 돌려 데이터가 20~25%미만인 데이터를 지우고 데이터를 재구성 한뒤
# 모델별로 결과 구하기

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)
#     alcohol malic_acid   ash alcalinity_of_ash magnesium  ... proanthocyanins color_intensity   hue od280/od315_of_diluted_wines proline
# 0     14.23       1.71  2.43              15.6     127.0  ...            2.29            5.64  1.04                         3.92  1065.0
# 1     13.20       1.78  2.14              11.2     100.0  ...            1.28            4.38  1.05                         3.40  1050.0
# 2     13.16       2.36  2.67              18.6     101.0  ...            2.81            5.68  1.03                         3.17  1185.0
# 3     14.37       1.95  2.50              16.8     113.0  ...            2.18            7.80  0.86                         3.45  1480.0
# 4     13.24       2.59  2.87              21.0     118.0  ...            1.82            4.32  1.04                         2.93   735.0
# ..      ...        ...   ...               ...       ...  ...             ...             ...   ...                          ...     ...
# 173   13.71       5.65  2.45              20.5      95.0  ...            1.06            7.70  0.64                         1.74   740.0
# 174   13.40       3.91  2.48              23.0     102.0  ...            1.41            7.30  0.70                         1.56   750.0
# 175   13.27       4.28  2.26              20.0     120.0  ...            1.35           10.20  0.59                         1.56   835.0
# 176   13.17       2.59  2.37              20.0     120.0  ...            1.46            9.30  0.60                         1.62   840.0
# 177   14.13       4.10  2.74              24.5      96.0  ...            1.35            9.20  0.61                         1.60   560.0

df.drop(['ash', 'nonflavanoid_phenols', 'proanthocyanins'], inplace=True, axis=1)
x = df.to_numpy()
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = DecisionTreeClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)

# RandomForestClassifier
# df.drop(['ash', 'nonflavanoid_phenols', 'proanthocyanins'], inplace=True, axis=1)
# acc :  1.0
# [0.12385265 0.02409247 0.03378788 0.03157412 0.0368482  0.17466158
#  0.16745269 0.07554236 0.14068513 0.19150292]

# GradientBoostingClassifier
# df.drop(['ash', 'nonflavanoid_phenols', 'proanthocyanins'], inplace=True, axis=1)
# acc :  1.0
# [2.60547665e-02 4.35617865e-02 1.35284718e-02 1.88013905e-03
#  7.24208941e-05 1.27714933e-01 2.60572658e-01 2.59516520e-02
#  2.20058347e-01 2.80604826e-01]

# DecisionTreeClassifier
# df.drop(['ash', 'nonflavanoid_phenols', 'proanthocyanins'], inplace=True, axis=1)
# acc :  0.9444444444444444
# [0.00489447 0.03045446 0.01598859 0.         0.         0.1569445
#  0.04078249 0.0555874  0.33215293 0.36319516]

# XGBClassifier
# df.drop(['ash', 'nonflavanoid_phenols', 'proanthocyanins'], inplace=True, axis=1)
# acc :  1.0
# [0.0240423  0.04775717 0.02414079 0.02944618 0.00902423 0.13611148
#  0.15712538 0.02496104 0.4071389  0.14025253]
