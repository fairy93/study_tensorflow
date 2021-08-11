import warnings
from numpy.lib.npyio import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_boston
from sklearn import datasets

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='regressor') #분류 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test,y_predict)
        print(model,'r2 : ',r2)
    except:
        print(name,'not found')
        continue
    
# 모델의 갯수 54
# ARDRegression() r2 :  0.734031745680535
# AdaBoostRegressor() r2 :  0.7585443689456308
# BaggingRegressor() r2 :  0.8238190471689657     
# BayesianRidge() r2 :  0.730275612907473
# CCA() r2 :  0.6872027154039331
# DecisionTreeRegressor() r2 :  0.5964062144926727
# DummyRegressor() r2 :  -0.007529979689363753    
# ElasticNet() r2 :  0.7017085897017612
# ElasticNetCV() r2 :  0.6828716996188751      
# ExtraTreeRegressor() r2 :  0.5466596121591991
# ExtraTreesRegressor() r2 :  0.8581023846409412     
# GammaRegressor() r2 :  -0.007529979689363753       
# GaussianProcessRegressor() r2 :  -6.524083110995924
# GradientBoostingRegressor() r2 :  0.8890717117463811
# HistGradientBoostingRegressor() r2 :  0.8729154797922107
# HuberRegressor() r2 :  0.6472721719827286
# IsotonicRegression not found
# KNeighborsRegressor() r2 :  0.5291986938774944
# KernelRidge() r2 :  0.702736992641771
# Lars() r2 :  0.7388195594672046
# LarsCV() r2 :  0.7383845499238595
# Lasso() r2 :  0.7004969973333021
# LassoCV() r2 :  0.7120057600938496
# LassoLars() r2 :  -0.007529979689363753
# LassoLarsCV() r2 :  0.7401218046842992
# LassoLarsIC() r2 :  0.7400000735850613
# LinearRegression() r2 :  0.7392832939303402
# LinearSVR() r2 :  0.4358847014013162
# MLPRegressor() r2 :  0.502298553546794
# MultiOutputRegressor not found
# MultiTaskElasticNet not found
# MultiTaskElasticNetCV not found
# MultiTaskLasso not found
# MultiTaskLassoCV not found
# NuSVR() r2 :  0.2250311891201786
# OrthogonalMatchingPursuit() r2 :  0.567862430332265
# OrthogonalMatchingPursuitCV() r2 :  0.6685983148788083
# PLSCanonical() r2 :  -2.998262060472451
# PLSRegression() r2 :  0.6983491930387439
# PassiveAggressiveRegressor() r2 :  -1.3458540978672504
# PoissonRegressor() r2 :  0.7458711880852642
# RANSACRegressor() r2 :  -0.10731125437625622
# RadiusNeighborsRegressor not found
# RandomForestRegressor() r2 :  0.8385501090301076
# RegressorChain not found
# Ridge() r2 :  0.7371871230992428
# RidgeCV(alphas=array([ 0.1,  1. , 10. ])) r2 :  0.7393085492641325
# SGDRegressor() r2 :  -2.1347953395062635e+26
# SVR() r2 :  0.20742855767899515
# StackingRegressor not found
# TheilSenRegressor(max_subpopulation=10000) r2 :  0.6955939867990673
# TransformedTargetRegressor() r2 :  0.7392832939303402
# TweedieRegressor() r2 :  0.6770019249379479
# VotingRegressor not found