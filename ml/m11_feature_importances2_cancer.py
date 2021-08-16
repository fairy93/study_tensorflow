import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

#2. 모델
model = DecisionTreeClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)
print('acc : ',acc)

# acc :  0.9210526315789473
# [0.         0.04272324 0.         0.         0.         0.
#  0.         0.08335616 0.         0.         0.         0.
#  0.         0.0121421  0.         0.02905547 0.         0.00860903 
#  0.         0.01742232 0.         0.03511622 0.70307139 0.
#  0.         0.00621763 0.         0.06228644 0.         0.        ]

print(model.feature_importances_)


def plot_feature_importances_dataset(modeel):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feautres")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()


