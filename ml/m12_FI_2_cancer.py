# feature_importance를 돌려 데이터가 20~25%미만인 데이터를 지우고 데이터를 재구성 한뒤
# 모델별로 결과 구하기

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)
#     mean radius mean texture mean perimeter mean area  ... worst concavity worst concave points worst symmetry worst fractal dimension
# 0         17.99        10.38         122.80    1001.0  ...          0.7119               0.2654         0.4601                 0.11890
# 1         20.57        17.77         132.90    1326.0  ...          0.2416               0.1860         0.2750                 0.08902
# 2         19.69        21.25         130.00    1203.0  ...          0.4504               0.2430         0.3613                 0.08758
# 3         11.42        20.38          77.58     386.1  ...          0.6869               0.2575         0.6638                 0.17300
# 4         20.29        14.34         135.10    1297.0  ...          0.4000               0.1625         0.2364                 0.07678
# ..          ...          ...            ...       ...  ...             ...                  ...            ...                     ...
# 564       21.56        22.39         142.00    1479.0  ...          0.4107               0.2216         0.2060                 0.07115
# 565       20.13        28.25         131.20    1261.0  ...          0.3215               0.1628         0.2572                 0.06637
# 566       16.60        28.08         108.30     858.1  ...          0.3403               0.1418         0.2218                 0.07820
# 567       20.60        29.33         140.10    1265.0  ...          0.9387               0.2650         0.4087                 0.12400
# 568        7.76        24.54          47.92     181.0  ...          0.0000               0.0000         0.2871                 0.07039

df.drop(['mean fractal dimension', 'smoothness error', 'compactness error',
         'concave points error', 'symmetry error', 'fractal dimension error',
         'mean symmetry', 'worst fractal dimension'], inplace=True, axis=1)
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
# df.drop(['mean fractal dimension','smoothness error','compactness error',
#  'concave points error','symmetry error','fractal dimension error',
# 'mean symmetry' ,'worst fractal dimension'], inplace=True, axis=1)
# acc :  0.9736842105263158
# [0.0516952  0.01876038 0.03722023 0.05069142 0.00658861 0.00976954
#  0.04255327 0.08401565 0.01445076 0.0050723  0.01416374 0.0456253
#  0.00668494 0.13593823 0.02143849 0.11774176 0.1432488  0.01099462
#  0.01350105 0.0379596  0.12435049 0.00753561]

# GradientBoostingClassifier
# df.drop(['mean fractal dimension','smoothness error','compactness error',
#  'concave points error','symmetry error','fractal dimension error',
# 'mean symmetry' ,'worst fractal dimension'], inplace=True, axis=1)
# acc :  0.9473684210526315
# [1.59671214e-04 3.72068662e-02 8.36966772e-04 2.61567965e-03
#  2.17915971e-03 1.05854605e-04 1.92850498e-03 1.26027205e-01
#  3.99626184e-03 3.09487245e-04 3.90076475e-04 1.79776989e-02
#  5.60591163e-03 3.31631323e-01 4.38202098e-02 4.26893154e-02
#  2.62997068e-01 4.68313821e-03 2.66116715e-04 1.44637186e-02
#  9.97901179e-02 3.19648974e-04]

# DecisionTreeClassifier
# df.drop(['mean fractal dimension','smoothness error','compactness error',
#  'concave points error','symmetry error','fractal dimension error',
# 'mean symmetry' ,'worst fractal dimension'], inplace=True, axis=1)
# acc :  0.9210526315789473
# [0.         0.05940707 0.         0.         0.         0.
#  0.         0.01967507 0.         0.         0.         0.02404987
#  0.01682964 0.         0.02236638 0.         0.71474329 0.
#  0.00624605 0.00461856 0.13206406 0.        ]

# XGBClassifier
# df.drop(['mean fractal dimension','smoothness error','compactness error',
#  'concave points error','symmetry error','fractal dimension error',
# 'mean symmetry' ,'worst fractal dimension'], inplace=True, axis=1)
# acc :  0.9736842105263158
# [0.00257534 0.05205284 0.00364253 0.01872069 0.00572702 0.01020269
#  0.00055066 0.1114669  0.00559427 0.00547231 0.01845575 0.01329992
#  0.00274876 0.01599405 0.02054472 0.30606112 0.31184995 0.00484172
#  0.00195511 0.00967049 0.07685644 0.00171671]
