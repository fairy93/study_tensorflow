import warnings
import numpy as np

from sklearn.model_selection import KFold,cross_val_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()

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

#모델의 갯수 41
# AdaBoostClassifier [0.96666667 0.93333333 1.         0.96666667 0.9       ] 0.9533
# BaggingClassifier [0.96666667 0.9        1.         1.         0.9       ] 0.9533
# BernoulliNB [0.26666667 0.26666667 0.3        0.13333333 0.2       ] 0.2333
# CalibratedClassifierCV [0.9        0.9        0.93333333 0.86666667 0.86666667] 0.8933
# CategoricalNB [0.93333333 0.9        0.96666667 0.9        0.93333333] 0.9267
# ClassifierChain not found
# ComplementNB [0.56666667 0.73333333 0.66666667 0.86666667 0.5       ] 0.6667
# DecisionTreeClassifier [0.96666667 0.93333333 1.         0.93333333 0.9       ] 0.9467
# DummyClassifier [0.26666667 0.26666667 0.3        0.13333333 0.2       ] 0.2333
# ExtraTreeClassifier [0.96666667 0.93333333 1.         0.96666667 0.93333333] 0.96
# ExtraTreesClassifier [0.96666667 0.9        1.         1.         0.9       ] 0.9533
# GaussianNB [0.93333333 0.9        1.         1.         0.9       ] 0.9467
# GaussianProcessClassifier [0.96666667 0.86666667 1.         0.96666667 0.9       ] 0.94
# GradientBoostingClassifier [0.96666667 0.9        1.         0.96666667 0.9       ] 0.9467
# HistGradientBoostingClassifier [0.96666667 0.9        1.         0.93333333 0.9       ] 0.94
# KNeighborsClassifier [0.96666667 0.96666667 1.         1.         0.9       ] 0.9667
# LabelPropagation [1.  0.9 1.  1.  0.9] 0.96
# LabelSpreading [1.  0.9 1.  1.  0.9] 0.96
# LinearDiscriminantAnalysis [1.         0.96666667 1.         1.         0.93333333] 0.98
# LinearSVC [0.93333333 0.9        1.         0.96666667 0.9       ] 0.94
# LogisticRegression [0.96666667 0.9        1.         1.         0.9       ] 0.9533
# LogisticRegressionCV [0.96666667 0.96666667 1.         1.         0.93333333] 0.9733
# MLPClassifier [0.96666667 0.96666667 1.         0.96666667 0.9       ] 0.96
# MultiOutputClassifier not found
# MultinomialNB [0.76666667 0.86666667 1.         0.56666667 0.56666667] 0.7533
# NearestCentroid [0.9        0.83333333 1.         0.96666667 0.9       ] 0.92
# NuSVC [0.96666667 0.86666667 1.         0.96666667 0.9       ] 0.94
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier [0.7        0.9        0.76666667 0.86666667 0.86666667] 0.82
# Perceptron [0.5        0.66666667 0.9        0.83333333 0.7       ] 0.72
# QuadraticDiscriminantAnalysis [1.         0.96666667 1.         1.         0.93333333] 0.98
# RadiusNeighborsClassifier [0.96666667 0.83333333 1.         0.93333333 0.93333333] 0.9333
# RandomForestClassifier [0.96666667 0.9        1.         0.96666667 0.9       ] 0.9467
# RidgeClassifier [0.76666667 0.83333333 0.86666667 0.86666667 0.8       ] 0.8267
# RidgeClassifierCV [0.76666667 0.83333333 0.86666667 0.86666667 0.8       ] 0.8267
# SGDClassifier [0.86666667 0.93333333 0.76666667 0.9        0.5       ] 0.7933
# SVC [0.96666667 0.86666667 1.         0.96666667 0.9       ] 0.94
# StackingClassifier not found
# VotingClassifier not found