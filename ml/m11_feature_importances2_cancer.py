import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data,datasets.target,train_size=0.8,random_state=79)

#2. 모델
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)
print('acc : ',acc)

# acc :  0.956140350877193
# [0.02422746 0.01427964 0.05897343 0.04093749 0.00527042 0.00828683
#  0.06062107 0.13156952 0.00501369 0.00459688 0.0157606  0.00609896
#  0.00999208 0.05105322 0.00456778 0.0045045  0.0067703  0.00287808
#  0.00364619 0.00569037 0.13484237 0.01682962 0.13976254 0.06372121
#  0.0108707  0.01698525 0.02212417 0.11777924 0.00672652 0.00561986]

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


