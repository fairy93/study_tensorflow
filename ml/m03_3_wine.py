from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=79)

#2. 모델
model = LinearSVC()
# model.score  0.8888888888888888
# acc_score  0.8888888888888888
# model =SVC()
# model.score  0.6481481481481481
# acc_score  0.6481481481481481
# model = KNeighborsClassifier()
# model.score  0.7222222222222222
# acc_score  0.7222222222222222
# model =LogisticRegression()
# model.score  0.9629629629629629
# acc_score  0.9629629629629629
# model = DecisionTreeClassifier()
# model.score  0.9814814814814815
# acc_score  0.9814814814814815
# model = RandomForestClassifier()
# model.score  1.0
# acc_score  1.0


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
results = model.score(x_test, y_test)
print('model.score : ', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc_score : ", acc)


