import warnings
from numpy.lib.npyio import load
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_diabetes
from sklearn import datasets
import numpy as np
warnings.filterwarnings('ignore')

#1 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='regressor') #분류 (type_filter ='regressor')
# print(allAlgorithms) 
kfold = KFold(n_splits=5, shuffle=True, random_state=66)

print('모델의 갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        score = cross_val_score(model, x,y, cv=kfold)
        print(name, ' r2 : ', np.round(np.mean(score), 4))
    except:
        print(name,'not found')
        continue

# 모델의 갯수 54
# ARDRegression  r2 :  0.4923
# AdaBoostRegressor  r2 :  0.4383
# BaggingRegressor  r2 :  0.3479
# BayesianRidge  r2 :  0.4893
# CCA  r2 :  0.438
# DecisionTreeRegressor  r2 :  -0.1536
# DummyRegressor  r2 :  -0.0033
# ElasticNet  r2 :  0.0054
# ElasticNetCV  r2 :  0.4394
# ExtraTreeRegressor  r2 :  -0.1676
# ExtraTreesRegressor  r2 :  0.4432
# GammaRegressor  r2 :  0.0027
# GaussianProcessRegressor  r2 :  -11.0753
# GradientBoostingRegressor  r2 :  0.438
# HistGradientBoostingRegressor  r2 :  0.3947
# HuberRegressor  r2 :  0.4822
# IsotonicRegression  r2 :  nan
# KNeighborsRegressor  r2 :  0.3673
# KernelRidge  r2 :  -3.5938
# Lars  r2 :  -0.1495
# LarsCV  r2 :  0.4879
# Lasso  r2 :  0.3518
# LassoCV  r2 :  0.487
# LassoLars  r2 :  0.3742
# LassoLarsCV  r2 :  0.4866
# LassoLarsIC  r2 :  0.4912
# LinearRegression  r2 :  0.4876
# LinearSVR  r2 :  -0.3702
# MLPRegressor  r2 :  -3.1072
# MultiOutputRegressor not found
# MultiTaskElasticNet  r2 :  nan
# MultiTaskElasticNetCV  r2 :  nan
# MultiTaskLasso  r2 :  nan
# MultiTaskLassoCV  r2 :  nan
# NuSVR  r2 :  0.1618
# OrthogonalMatchingPursuit  r2 :  0.3121
# OrthogonalMatchingPursuitCV  r2 :  0.4857
# PLSCanonical  r2 :  -1.2086
# PLSRegression  r2 :  0.4842
# PassiveAggressiveRegressor  r2 :  0.4612
# PoissonRegressor  r2 :  0.3341
# RANSACRegressor  r2 :  0.1192
# RadiusNeighborsRegressor  r2 :  -0.0033
# RandomForestRegressor  r2 :  0.4242
# RegressorChain not found
# Ridge  r2 :  0.4212
# RidgeCV  r2 :  0.4884
# SGDRegressor  r2 :  0.4089
# SVR  r2 :  0.1591
# StackingRegressor not found
# TheilSenRegressor  r2 :  0.48
# TransformedTargetRegressor  r2 :  0.4876
# TweedieRegressor  r2 :  0.0032
# VotingRegressor not found