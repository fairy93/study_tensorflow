# 실습
#1. 상단모델에 그리드/랜덤 서치로 튜닝한 모델 구성 최적의 r2값과 피쳐임포턴스 구하기
#2. 위 스레드값으로 SelectFromModel 돌려서 최적의 피처 갯수 구하기
#3. 위 피쳐 갯수로 피쳐 갯수 조정한뒤 다시 랜덤/그리드 서치 후 최적의 r2구하기
#4. 1vs3 비교

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split,GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
x = df[['s4','s5','s6','bmi']]

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

#2. 모델
parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        "tree_method":['gpu_hist'],
        "gpu_id":[0]
        }
# model = GridSearchCV(XGBRegressor(), parameters,cv=kfold, verbose=1)
model =XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=5, gpu_id=0,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=4,
             min_child_weight=10, missing=np.nan, monotone_constraints='()',
             n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1.0,
             tree_method='gpu_hist', validate_parameters=1, verbosity=None)

#3. 훈련
model.fit(x_train, y_train) 

#4. 평가 예측
score = model.score(x_test, y_test)
print('r2 :', score)
# print('best_estimator_ : ', model.best_estimator_)
# print('best_score_  :', model.best_score_)

# def plot_feature_importance_dataset(model):
#       n_features = datasets.data.shape[1]
#       plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#       plt.yticks(np.arange(n_features), datasets.feature_names)
#       plt.xlabel("Feature Importances")
#       plt.ylabel("Features")
#       plt.ylim(-1, n_features)

# plot_feature_importance_dataset(model)
# plt.show()

# thresholds = np.sort(model.feature_importances_)
# print(thresholds)
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)

#     seletion_model = XGBRegressor(n_jobs=-1)
#     seletion_model.fit(select_x_train, y_train)

#     y_pred = seletion_model.predict(select_x_test)

#     score = r2_score(y_test, y_pred)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))

# result1
# r2 : 0.3724407370798575
# Best estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.6, gamma=5, gpu_id=0,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=4,
#              min_child_weight=10, missing=nan, monotone_constraints='()',
#              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1.0,
#              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
# best_score_  : 0.346810005041083

# result2
# Thresh=0.032, n=10, R2: 41.41%
# Thresh=0.048, n=9, R2: 31.94%
# Thresh=0.054, n=8, R2: 38.53%
# Thresh=0.058, n=7, R2: 36.95%
# Thresh=0.060, n=6, R2: 36.22%
# Thresh=0.072, n=5, R2: 48.75%
# Thresh=0.090, n=4, R2: 42.08%
# Thresh=0.155, n=3, R2: 35.29%
# Thresh=0.209, n=2, R2: 34.80%
# Thresh=0.220, n=1, R2: 20.53%

# result3
# r2 : 0.41900951847324575

# result4
# result3 의 결과 가더좋음