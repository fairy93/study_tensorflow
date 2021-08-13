from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

#2. 모델
# model = LinearSVC()
# model.score  0.8888888888888888
# acc_score  0.8888888888888888
# model =SVC()
# model.score  0.9122807017543859
# acc_score  0.9122807017543859
# model = KNeighborsClassifier()
# model.score  0.9239766081871345
# acc_score  0.9239766081871345
# model =LogisticRegression()
# model.score  0.9122807017543859
# acc_score  0.9122807017543859
# model = DecisionTreeClassifier()
# model.score  0.9532163742690059
# acc_score  0.9532163742690059
model = RandomForestClassifier()
# model.score  0.9649122807017544
# acc_score  0.9649122807017544


#3. 컴파일구현
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)
