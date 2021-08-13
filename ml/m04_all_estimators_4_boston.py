import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

#2. 모델
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms))
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
# ARDRegression() r2 :  0.6944696184182546
# AdaBoostRegressor() r2 :  0.8094855780224868
# BaggingRegressor() r2 :  0.8034488159551316
# BayesianRidge() r2 :  0.695856508140514
# CCA() r2 :  0.6442219855253039
# DecisionTreeRegressor() r2 :  0.5173013468280532
# DummyRegressor() r2 :  -0.0041068958105112685
# ElasticNet() r2 :  0.6721234053512837
# ElasticNetCV() r2 :  0.6553448317204869
# ExtraTreeRegressor() r2 :  0.39685269617987695
# ExtraTreesRegressor() r2 :  0.8395154280783148
# GammaRegressor() r2 :  -0.0041068958105112685
# GaussianProcessRegressor() r2 :  -7.590783910607163
# GradientBoostingRegressor() r2 :  0.8766714701063418
# HistGradientBoostingRegressor() r2 :  0.869269988717646
# HuberRegressor() r2 :  0.6238684200133697
# IsotonicRegression not found
# KNeighborsRegressor() r2 :  0.5078560102508338
# KernelRidge() r2 :  0.6564382009516103
# Lars() r2 :  0.65631981343613
# LarsCV() r2 :  0.6596339897864862
# Lasso() r2 :  0.6674225172456512
# LassoCV() r2 :  0.6789710961144446
# LassoLars() r2 :  -0.0041068958105112685
# LassoLarsCV() r2 :  0.6992648403291186
# LassoLarsIC() r2 :  0.6985355671699824
# LinearRegression() r2 :  0.699531931256025
# LinearSVR() r2 :  0.6215685446625165
# MLPRegressor() r2 :  0.4385164673780121
# MultiOutputRegressor not found
# MultiTaskElasticNet not found
# MultiTaskElasticNetCV not found
# MultiTaskLasso not found
# MultiTaskLassoCV not found
# NuSVR() r2 :  0.24154568510581298
# OrthogonalMatchingPursuit() r2 :  0.5642740976259404
# OrthogonalMatchingPursuitCV() r2 :  0.6287011255056013
# PLSCanonical() r2 :  -3.4092059271475756
# PLSRegression() r2 :  0.6358282107815543
# PassiveAggressiveRegressor() r2 :  0.19331590172538993
# PoissonRegressor() r2 :  0.7418559076228988
# RANSACRegressor() r2 :  0.40092827817692445
# RadiusNeighborsRegressor not found
# RandomForestRegressor() r2 :  0.8503813590899717
# RegressorChain not found
# Ridge() r2 :  0.6997720678747266
# RidgeCV(alphas=array([ 0.1,  1. , 10. ])) r2 :  0.6999493623176783
# SGDRegressor() r2 :  -7.926382821690771e+25
# SVR() r2 :  0.22139963324298728
# StackingRegressor not found
# TheilSenRegressor(max_subpopulation=10000) r2 :  0.665219872258659
# TransformedTargetRegressor() r2 :  0.699531931256025
# TweedieRegressor() r2 :  0.6487454754103463
# VotingRegressor not found