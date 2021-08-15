# 피쳐 임포턴스가 전체 중요도에서 하위 20 ~ 25% 컬럼들을 제거 하여 데이터
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
datasets = load_iris()
# print(type(datasets)) # <class 'sklearn.utils.Bunch'>
x = datasets.data
y = datasets.target
print(type(x)) # <class 'numpy.ndarray'>
print(type(y)) # <class 'numpy.ndarray'>
print(x)

# x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

# #2. 모델
# model = DecisionTreeClassifier()
# # model = RandomForestClassifier()
# # model = XGBClassifier()
# # model = GradientBoostingClassifier()

# #3. 훈련
# model.fit(x_train,y_train)

# #4. 평가 예측
# acc = model.score(x_test,y_test)
# print('acc : ',acc)

# print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_feature_importances_dataset(modeel):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Feautres")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()


