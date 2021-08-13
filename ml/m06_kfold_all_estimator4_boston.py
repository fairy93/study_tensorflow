import warnings
import numpy as np

from sklearn.model_selection import KFold,cross_val_score
from sklearn.datasets import load_boston
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

#2. 모델
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
# AdaBoostRegressor  r2 :  0.8385
# BaggingRegressor  r2 :  0.8796
# BayesianRidge  r2 :  0.7038
# CCA  r2 :  0.6471
# DecisionTreeRegressor  r2 :  0.7275
# DummyRegressor  r2 :  -0.0135
# ElasticNet  r2 :  0.6708
# ElasticNetCV  r2 :  0.6565
# ExtraTreeRegressor  r2 :  0.6445
# ExtraTreesRegressor  r2 :  0.8775
# GammaRegressor  r2 :  -0.0136
# GaussianProcessRegressor  r2 :  -5.9286
# GradientBoostingRegressor  r2 :  0.8843
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
# LinearSVR  r2 :  0.3087
# MLPRegressor  r2 :  0.41
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
# PassiveAggressiveRegressor  r2 :  0.061
# PoissonRegressor  r2 :  0.7549
# RANSACRegressor  r2 :  -0.0451
# RadiusNeighborsRegressor  r2 :  nan
# RandomForestRegressor  r2 :  0.8779
# RegressorChain not found
# Ridge  r2 :  0.7109
# RidgeCV  r2 :  0.7128
# SGDRegressor  r2 :  -1.1218531683939522e+26
# SVR  r2 :  0.1963
# StackingRegressor not found
# TheilSenRegressor  r2 :  0.6736
# TransformedTargetRegressor  r2 :  0.7128
# TweedieRegressor  r2 :  0.6558
# VotingRegressor not found