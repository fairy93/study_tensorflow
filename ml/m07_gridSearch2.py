import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

Kfold = KFold(n_splits=5, shuffle=True, random_state=66)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=70)


parameters = {
    'kernel':  ['sigmoid','linear','rbf'],
    'gamma': [1000,100,10, 1, 0, 0.1, 0.01, 0.001,0.0001],
    'C':[1000,100, 10, 10, 0, 0.1, 0.01, 0.001,0.0001]
}
#2. 모델
model = SVC(C=1, kernel='linear')

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
print("model.scored : ", model.score(x_test,y_test)) # =accuracy_score

y_pred = model.predict(x_test)
print("accuracy_score : ",accuracy_score(y_test,y_pred))

# model.scored :  1.0
# accuracy_score :  1.0