import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

#1. 데이터
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
model = XGBRegressor(n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
score = model.score(x_train, y_train)
print(score)

thresholds = np.sort(model.feature_importances_)

# 0.9999895688923817
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    seletion_model = XGBRegressor(n_jobs=-1)
    seletion_model.fit(select_x_train, y_train)

    y_pred = seletion_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))

# Thresh=0.001, n=13, R2: 92.21%
# Thresh=0.004, n=12, R2: 92.16%
# Thresh=0.012, n=11, R2: 92.03%
# Thresh=0.012, n=10, R2: 92.19%
# Thresh=0.014, n=9, R2: 93.08%
# Thresh=0.015, n=8, R2: 92.37%
# Thresh=0.018, n=7, R2: 91.48%
# Thresh=0.030, n=6, R2: 92.71%
# Thresh=0.042, n=5, R2: 91.74%
# Thresh=0.052, n=4, R2: 92.11%
# Thresh=0.069, n=3, R2: 92.52%
# Thresh=0.301, n=2, R2: 69.41%
# Thresh=0.428, n=1, R2: 44.98%