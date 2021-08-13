import warnings
import numpy as np

from sklearn.model_selection import KFold,cross_val_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

#2. 모델
allAlgorithms = all_estimators(type_filter='classifier') #분류// # 회귀 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms)) # 41
kfold = KFold(n_splits=5, shuffle=True, random_state=79)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model,x,y,cv=kfold)
        print(name, scores, round(np.mean(scores), 4))
    except:
        print(name,'not found')
        continue

# 모델의 갯수 41
# AdaBoostClassifier [0.97368421 0.97368421 0.93859649 0.97368421 0.98230088] 0.9684
# BaggingClassifier [0.95614035 0.93859649 0.93859649 0.98245614 0.94690265] 0.9525
# BernoulliNB [0.65789474 0.63157895 0.57017544 0.5877193  0.69026549] 0.6275
# CalibratedClassifierCV [0.92105263 0.9122807  0.9122807  0.92105263 0.95575221] 0.9245
# CategoricalNB [       nan 0.90350877        nan        nan        nan] nan
# ClassifierChain not found
# ComplementNB [0.89473684 0.88596491 0.88596491 0.89473684 0.92035398] 0.8964
# DecisionTreeClassifier [0.92982456 0.95614035 0.95614035 0.92982456 0.95575221] 0.9455
# DummyClassifier [0.65789474 0.63157895 0.57017544 0.5877193  0.69026549] 0.6275
# ExtraTreeClassifier [0.94736842 0.94736842 0.95614035 0.93859649 0.92035398] 0.942
# ExtraTreesClassifier [0.96491228 0.94736842 0.96491228 0.97368421 0.96460177] 0.9631
# GaussianNB [0.96491228 0.90350877 0.93859649 0.93859649 0.94690265] 0.9385
# GaussianProcessClassifier [0.92105263 0.89473684 0.9122807  0.93859649 0.94690265] 0.9227
# GradientBoostingClassifier [0.96491228 0.95614035 0.97368421 0.95614035 0.95575221] 0.9613
# HistGradientBoostingClassifier [0.97368421 0.97368421 0.96491228 0.97368421 0.97345133] 0.9719
# KNeighborsClassifier [0.93859649 0.94736842 0.92105263 0.92105263 0.9380531 ] 0.9332
# LabelPropagation [0.36842105 0.39473684 0.45614035 0.43859649 0.30973451] 0.3935
# LabelSpreading [0.36842105 0.39473684 0.45614035 0.43859649 0.30973451] 0.3935
# LinearDiscriminantAnalysis [0.94736842 0.97368421 0.94736842 0.96491228 0.97345133] 0.9614
# LinearSVC [0.92105263 0.89473684 0.92105263 0.92105263 0.92920354] 0.9174
# LogisticRegression [0.93859649 0.93859649 0.94736842 0.93859649 0.92920354] 0.9385
# LogisticRegressionCV [0.97368421 0.95614035 0.94736842 0.97368421 0.96460177] 0.9631
# MLPClassifier [0.92982456 0.92982456 0.90350877 0.94736842 0.96460177] 0.935
# MultiOutputClassifier not found
# MultinomialNB [0.89473684 0.87719298 0.89473684 0.89473684 0.92035398] 0.8964
# NearestCentroid [0.89473684 0.88596491 0.90350877 0.87719298 0.89380531] 0.891
# NuSVC [0.89473684 0.85087719 0.88596491 0.85964912 0.87610619] 0.8735
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier [0.94736842 0.89473684 0.70175439 0.89473684 0.94690265] 0.8771
# Perceptron [0.76315789 0.9122807  0.89473684 0.9122807  0.87610619] 0.8717
# QuadraticDiscriminantAnalysis [0.92982456 0.94736842 0.95614035 0.98245614 0.96460177] 0.9561
# RadiusNeighborsClassifier [nan nan nan nan nan] nan
# RandomForestClassifier [0.95614035 0.93859649 0.95614035 0.98245614 0.96460177] 0.9596
# RidgeClassifier [0.95614035 0.95614035 0.95614035 0.95614035 0.96460177] 0.9578
# RidgeClassifierCV [0.96491228 0.95614035 0.95614035 0.97368421 0.96460177] 0.9631
# SGDClassifier [0.90350877 0.89473684 0.88596491 0.92105263 0.94690265] 0.9104
# SVC [0.92105263 0.90350877 0.9122807  0.92982456 0.92920354] 0.9192
# StackingClassifier not found
# VotingClassifier not found