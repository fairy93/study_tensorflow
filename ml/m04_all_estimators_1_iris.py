import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2, shuffle=True, random_state=79)

#2. 모델
allAlgorithms = all_estimators(type_filter='classifier') # 분류 / # 회귀 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(model,'acc : ',acc)
    except:
        print(name,'not found')
        continue

# 모델의 갯수 41
# AdaBoostClassifier() acc :  1.0
# BaggingClassifier() acc :  1.0
# BernoulliNB() acc :  0.0
# CalibratedClassifierCV() acc :  1.0
# CategoricalNB() acc :  1.0
# ClassifierChain not found
# ComplementNB() acc :  1.0
# DecisionTreeClassifier() acc :  1.0
# DummyClassifier() acc :  0.0
# ExtraTreeClassifier() acc :  1.0
# ExtraTreesClassifier() acc :  1.0
# GaussianNB() acc :  1.0
# GaussianProcessClassifier() acc :  1.0
# GradientBoostingClassifier() acc :  1.0
# HistGradientBoostingClassifier() acc :  1.0
# KNeighborsClassifier() acc :  1.0
# LabelPropagation() acc :  1.0
# LabelSpreading() acc :  1.0
# LinearDiscriminantAnalysis() acc :  1.0
# LinearSVC() acc :  1.0
# LogisticRegression() acc :  1.0
# LogisticRegressionCV() acc :  1.0
# MLPClassifier() acc :  1.0
# MultiOutputClassifier not found
# MultinomialNB() acc :  1.0
# NearestCentroid() acc :  1.0
# NuSVC() acc :  1.0
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier() acc :  1.0
# Perceptron() acc :  1.0
# QuadraticDiscriminantAnalysis() acc :  1.0
# RadiusNeighborsClassifier() acc :  1.0
# RandomForestClassifier() acc :  1.0
# RidgeClassifier() acc :  1.0
# RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])) acc :  1.0
# SGDClassifier() acc :  1.0
# SVC() acc :  1.0
# StackingClassifier not found
# VotingClassifier not found