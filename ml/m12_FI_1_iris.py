# feature_importance를 돌려 데이터가 20~25%미만인 데이터를 지우고 데이터를 재구성 한뒤
# 모델별로 결과 구하기

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)

#     sepal length (cm) sepal width (cm) petal length (cm) petal width (cm)
# 0                 5.1              3.5               1.4              0.2
# 1                 4.9              3.0               1.4              0.2
# 2                 4.7              3.2               1.3              0.2
# 3                 4.6              3.1               1.5              0.2
# 4                 5.0              3.6               1.4              0.2
# ..                ...              ...               ...              ...
# 145               6.7              3.0               5.2              2.3
# 146               6.3              2.5               5.0              1.9
# 147               6.5              3.0               5.2              2.0
# 148               6.2              3.4               5.4              2.3
# 149               5.9              3.0               5.1              1.8

df.drop(['sepal width (cm)', 'sepal length (cm)'], inplace=True, axis=1)
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
# df.drop(['sepal width (cm)', 'sepal length (cm)'], inplace=True, axis=1)
# acc :  0.9666666666666667
# [0.51092421 0.48907579]

# GradientBoostingClassifier
# df.drop(['sepal width (cm)', 'sepal length (cm)'], inplace=True, axis=1)
# acc :  0.9666666666666667
# [0.29559522 0.70440478]

# DecisionTreeClassifier
# df.drop(['sepal width (cm)', 'sepal length (cm)'], inplace=True, axis=1)
# acc :  0.9333333333333333
# [0.54517411 0.45482589]

# XGBClassifier
# df.drop(['sepal width (cm)', 'sepal length (cm)'], inplace=True, axis=1)
# acc :  0.9666666666666667
# [0.510896   0.48910394]
