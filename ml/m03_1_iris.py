from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

#2. 모델
# model = LinearSVC()
# acc_score  0.9555555555555556
# model =SVC()
# acc_score  0.9555555555555556
# model = KNeighborsClassifier()
# acc_score  0.9777777777777777
# model =LogisticRegression()
# acc_score  0.9555555555555556
# model = DecisionTreeClassifier()
# acc_score  0.9777777777777777
model = RandomForestClassifier()
# acc_score  0.9555555555555556

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score ', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc_score : ", acc)

print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2)
