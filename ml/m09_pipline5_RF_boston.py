import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

#2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("model.scored : ", model.score(x_test,y_test)) # =accuracy_score

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test,y_predict))

# model.scored :  0.9179066153388223
# r2_score :  0.9179066153388223