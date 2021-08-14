import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("model.scored : ", model.score(x_test,y_test)) # =accuracy_score

y_predict = model.predict(x_test)
print("accuracy_score : ",accuracy_score(y_test,y_predict))

# model.scored :  0.9555555555555556
# accuracy_score :  0.9555555555555556