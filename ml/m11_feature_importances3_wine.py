import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

#2. 모델
model = DecisionTreeClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)
print('acc : ',acc)

# acc :  0.9166666666666666
# [0.         0.         0.         0.040539   0.         0.
#  0.41901935 0.         0.         0.37913911 0.         0.
#  0.16130254]


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


