import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_diabetes

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

#2. 모델
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) 
print('모델의 갯수', len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        print(model,'r2 : ',r2)
    except:
        print(name,'not found')
        continue

# 모델의 갯수 54
# ARDRegression() r2 :  0.47791116522454236
# AdaBoostRegressor() r2 :  0.3817071952186434
# BaggingRegressor() r2 :  0.3997054632245669
# BayesianRidge() r2 :  0.4780529986175207
# CCA() r2 :  0.41107022944366156
# DecisionTreeRegressor() r2 :  0.005043582094847898
# DummyRegressor() r2 :  -0.005189786172508981
# ElasticNet() r2 :  0.002753327375223491
# ElasticNetCV() r2 :  0.42374870317352575
# ExtraTreeRegressor() r2 :  -0.30179576533776387
# ExtraTreesRegressor() r2 :  0.3923946257996228
# GammaRegressor() r2 :  0.0006404924784673138
# GaussianProcessRegressor() r2 :  -14.165392897652543
# GradientBoostingRegressor() r2 :  0.39717111574418784
# HistGradientBoostingRegressor() r2 :  0.373599763767692
# HuberRegressor() r2 :  0.4704831698022315
# IsotonicRegression not found
# KNeighborsRegressor() r2 :  0.33451669281767926
# KernelRidge() r2 :  -2.8622383860245675
# Lars() r2 :  -0.6678769459808147
# LarsCV() r2 :  0.4810280667579363
# Lasso() r2 :  0.34207210434388546
# LassoCV() r2 :  0.47630214113772085
# LassoLars() r2 :  0.36947512718608966
# LassoLarsCV() r2 :  0.47652405278954757
# LassoLarsIC() r2 :  0.4799276085787165
# LinearRegression() r2 :  0.4816266308783952
# LinearSVR() r2 :  -0.2361470285970797
# MLPRegressor() r2 :  -2.5721401126486167
# MultiOutputRegressor not found
# MultiTaskElasticNet not found
# MultiTaskElasticNetCV not found
# MultiTaskLasso not found
# MultiTaskLassoCV not found
# NuSVR() r2 :  0.1506766002075719
# OrthogonalMatchingPursuit() r2 :  0.20209905202375633
# OrthogonalMatchingPursuitCV() r2 :  0.4574140160905098
# PLSCanonical() r2 :  -0.9821719135708051
# PLSRegression() r2 :  0.48336926192877383
# PassiveAggressiveRegressor() r2 :  0.4624342265413762
# PoissonRegressor() r2 :  0.317446540276469
# RANSACRegressor() r2 :  0.06323441670109953
# RadiusNeighborsRegressor() r2 :  -0.005189786172508981
# RandomForestRegressor() r2 :  0.3916881992359862
# RegressorChain not found
# Ridge() r2 :  0.3988720823743557
# RidgeCV(alphas=array([ 0.1,  1. , 10. ])) r2 :  0.47982173779502635
# SGDRegressor() r2 :  0.36960305056229203
# SVR() r2 :  0.1724447664042561
# StackingRegressor not found
# TheilSenRegressor(max_subpopulation=10000) r2 :  0.4640463338000984
# TransformedTargetRegressor() r2 :  0.4816266308783952
# TweedieRegressor() r2 :  0.0004073741052421642
# VotingRegressor not found