from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC
from sklearn import datasets

#1 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델
model = LinearSVC()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)  # =acc
print('model.score : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2)

# model.score :  0.9555555555555556
# acc_score :  0.9555555555555556
# [0 2 1 1 1]
# [0 2 1 2 1]