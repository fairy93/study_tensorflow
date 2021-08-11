import warnings
from numpy.lib.npyio import load
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_boston
from sklearn import datasets
import numpy as np
warnings.filterwarnings('ignore')

#1 데이터
datasets = load_boston()

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
# ARDRegression  r2 :  0.6985
# AdaBoostRegressor  r2 :  0.8388
# BaggingRegressor  r2 :  0.8702
# BayesianRidge  r2 :  0.7038
# CCA  r2 :  0.6471
# DecisionTreeRegressor  r2 :  0.7666
# DummyRegressor  r2 :  -0.0135
# ElasticNet  r2 :  0.6708
# ElasticNetCV  r2 :  0.6565
# ExtraTreeRegressor  r2 :  0.7321
# ExtraTreesRegressor  r2 :  0.8779
# GammaRegressor  r2 :  -0.0136
# GaussianProcessRegressor  r2 :  -5.9286
# GradientBoostingRegressor  r2 :  0.8842
# HistGradientBoostingRegressor  r2 :  0.8581
# HuberRegressor  r2 :  0.584
# IsotonicRegression  r2 :  nan
# KNeighborsRegressor  r2 :  0.5286
# KernelRidge  r2 :  0.6854
# Lars  r2 :  0.6977
# LarsCV  r2 :  0.6928
# Lasso  r2 :  0.6657
# LassoCV  r2 :  0.6779
# LassoLars  r2 :  -0.0135
# LassoLarsCV  r2 :  0.6965
# LassoLarsIC  r2 :  0.713
# LinearRegression  r2 :  0.7128
# LinearSVR  r2 :  0.568
# MLPRegressor  r2 :  0.4511
# MultiOutputRegressor not found
# MultiTaskElasticNet  r2 :  nan
# MultiTaskElasticNetCV  r2 :  nan
# MultiTaskLasso  r2 :  nan
# MultiTaskLassoCV  r2 :  nan
# NuSVR  r2 :  0.2295
# OrthogonalMatchingPursuit  r2 :  0.5343
# OrthogonalMatchingPursuitCV  r2 :  0.6578
# PLSCanonical  r2 :  -2.2096
# PLSRegression  r2 :  0.6847
# PassiveAggressiveRegressor  r2 :  -1.1376
# PoissonRegressor  r2 :  0.7549
# RANSACRegressor  r2 :  0.3675
# RadiusNeighborsRegressor  r2 :  nan
# RandomForestRegressor  r2 :  0.8789
# RegressorChain not found
# Ridge  r2 :  0.7109
# RidgeCV  r2 :  0.7128
# SGDRegressor  r2 :  -2.720144045636607e+26
# SVR  r2 :  0.1963
# StackingRegressor not found
# TheilSenRegressor  r2 :  0.6714
# TransformedTargetRegressor  r2 :  0.7128
# TweedieRegressor  r2 :  0.6558
# VotingRegressor not found