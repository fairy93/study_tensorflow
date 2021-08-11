import warnings
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
from sklearn import datasets
import numpy as np
warnings.filterwarnings('ignore')

#1 데이터
datasets = load_iris()

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
# AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 0.8867
# BaggingClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667] 0.9467
# BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 0.2933
# CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 0.9133
# CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 0.9333
# ClassifierChain not found
# ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 0.6667
# DecisionTreeClassifier [0.93333333 0.96666667 1.         0.9        0.93333333] 0.9467
# DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 0.2933
# ExtraTreeClassifier [0.9        0.93333333 1.         0.9        0.93333333] 0.9333
# ExtraTreesClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667] 0.9467
# GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 0.9467
# GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
# GradientBoostingClassifier [0.93333333 0.96666667 1.         0.93333333 0.96666667] 0.96
# HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 0.94
# KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
# LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 0.96
# LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 0.96
# LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 0.98
# LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667
# LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 0.9667
# LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 0.9733
# MLPClassifier [0.96666667 1.         1.         0.93333333 1.        ] 0.98
# MultiOutputClassifier not found
# MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 0.9667
# NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 0.9333
# NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 0.9733
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier [0.93333333 0.96666667 0.73333333 0.7        0.96666667] 0.86
# Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 0.78
# QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ] 0.98
# RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ] 0.9533
# RandomForestClassifier [0.96666667 0.96666667 1.         0.86666667 0.96666667] 0.9533
# RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 0.84
# RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 0.84
# SGDClassifier [0.9        0.63333333 0.96666667 0.63333333 0.96666667] 0.82
# SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667
# StackingClassifier not found
# VotingClassifier not found