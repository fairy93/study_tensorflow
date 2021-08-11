import warnings
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
import numpy as np
warnings.filterwarnings('ignore')

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier') #분류// # 회귀 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms)) # 41
kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model,x,y,cv=kfold)
        print(name, scores, round(np.mean(scores), 4))
    except:
        print(name,'not found')
        continue

# 모델의 갯수 41
# AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 0.9649
# BaggingClassifier [0.94736842 0.92982456 0.97368421 0.93859649 0.95575221] 0.949
# BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 0.6274
# CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 0.9263
# CategoricalNB [nan nan nan nan nan] nan
# ClassifierChain not found
# ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 0.8963
# DecisionTreeClassifier [0.93859649 0.92105263 0.92982456 0.89473684 0.9380531 ] 0.9245
# DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 0.6274
# ExtraTreeClassifier [0.9122807  0.92105263 0.90350877 0.92982456 0.9380531 ] 0.9209
# ExtraTreesClassifier [0.95614035 0.96491228 0.96491228 0.94736842 0.99115044] 0.9649
# GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 0.942
# GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 0.9122
# GradientBoostingClassifier [0.94736842 0.97368421 0.95614035 0.94736842 0.97345133] 0.9596
# HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 0.9737
# KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928
# LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 0.3902
# LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 0.3902
# LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 0.9614
# LinearSVC [0.92105263 0.93859649 0.89473684 0.64035088 0.94690265] 0.8683
# LogisticRegression [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 0.9385
# LogisticRegressionCV [0.96491228 0.97368421 0.92105263 0.96491228 0.96460177] 0.9578
# MLPClassifier [0.89473684 0.94736842 0.9122807  0.9122807  0.94690265] 0.9227
# MultiOutputClassifier not found
# MultinomialNB [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 0.8928
# NearestCentroid [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 0.8893
# NuSVC [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 0.8735
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier [0.89473684 0.78070175 0.86842105 0.77192982 0.88495575] 0.8401
# Perceptron [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 0.7771
# QuadraticDiscriminantAnalysis [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 0.9525
# RadiusNeighborsClassifier [nan nan nan nan nan] nan
# RandomForestClassifier [0.97368421 0.96491228 0.97368421 0.93859649 0.97345133] 0.9649
# RidgeClassifier [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 0.9543
# RidgeClassifierCV [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 0.9561
# SGDClassifier [0.87719298 0.94736842 0.88596491 0.85964912 0.79646018] 0.8733
# SVC [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 0.921
# StackingClassifier not found
# VotingClassifier not found