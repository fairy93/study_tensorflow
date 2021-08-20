
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBClassifier

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model =XGBClassifier(n_estimators=100,learning_rate=0.15,n_jobs=1)

#3. 훈련
model.fit(x_train, y_train,verbose=1,eval_metric=['error', 'mae'],eval_set=[(x_train,y_train),(x_test,y_test)])

#4. 평가
results = model.score(x_test,y_test)
print('results : ', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('acc : ',acc)


print("==========================================")
hist = model.evals_result()
print(hist)

epochs = len(hist['validation_0']['error'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['error'], label='Train')
ax.plot(x_axis, hist['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost error')

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['mae'], label='Train')
ax.plot(x_axis, hist['validation_1']['mae'], label='Test')
ax.legend()
plt.ylabel('mae')
plt.title('XGBoost mae')
plt.show()