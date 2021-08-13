import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

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
# AdaBoostClassifier() acc :  0.9722222222222222
# BaggingClassifier() acc :  0.9722222222222222
# BernoulliNB() acc :  0.4722222222222222
# CalibratedClassifierCV() acc :  0.9722222222222222
# CategoricalNB not found
# ClassifierChain not found
# ComplementNB() acc :  0.7777777777777778
# DecisionTreeClassifier() acc :  0.9444444444444444
# DummyClassifier() acc :  0.4722222222222222
# ExtraTreeClassifier() acc :  0.8611111111111112
# ExtraTreesClassifier() acc :  1.0
# GaussianNB() acc :  1.0
# GaussianProcessClassifier() acc :  0.5277777777777778
# GradientBoostingClassifier() acc :  0.9722222222222222
# HistGradientBoostingClassifier() acc :  0.9444444444444444
# KNeighborsClassifier() acc :  0.7222222222222222
# LabelPropagation() acc :  0.5277777777777778
# LabelSpreading() acc :  0.5277777777777778
# LinearDiscriminantAnalysis() acc :  1.0
# LinearSVC() acc :  0.9444444444444444
# LogisticRegression() acc :  1.0
# LogisticRegressionCV() acc :  1.0
# MLPClassifier() acc :  0.5277777777777778
# MultiOutputClassifier not found
# MultinomialNB() acc :  0.8611111111111112
# NearestCentroid() acc :  0.7222222222222222
# NuSVC() acc :  0.9722222222222222
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier() acc :  0.3333333333333333
# Perceptron() acc :  0.6388888888888888
# QuadraticDiscriminantAnalysis() acc :  0.9722222222222222
# RadiusNeighborsClassifier not found
# RandomForestClassifier() acc :  1.0
# RidgeClassifier() acc :  1.0
# RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])) acc :  1.0
# SGDClassifier() acc :  0.6944444444444444
# SVC() acc :  0.75
# StackingClassifier not found
# VotingClassifier not found